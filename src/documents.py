from functools import partial
import hashlib
import io
import logging
import os
from pathlib import Path
import re
from typing import ByteString, Callable, List, Optional, Tuple, Union
import uuid

from charset_normalizer import from_bytes
from nougat import NougatModel
from nougat.dataset.rasterize import rasterize_paper
from nougat.utils.dataset import LazyDataset
from nougat.utils.device import move_to_device
from nougat.utils.checkpoint import get_checkpoint
from nougat.postprocessing import markdown_compatible
import pymupdf
import pymupdf4llm
import pypdfium2
from omegaconf import DictConfig
import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm


def rasterize_paper(
    pdf: Union[Path, bytes],
    outpath: Optional[Path] = None,
    dpi: int = 96,
    return_pil=False,
    pages=None,
) -> Optional[List[io.BytesIO]]:
    """
    Rasterize a PDF file to PNG images.

    Args:
        pdf (Path): The path to the PDF file.
        outpath (Optional[Path], optional): The output directory. If None, the PIL images will be returned instead. Defaults to None.
        dpi (int, optional): The output DPI. Defaults to 96.
        return_pil (bool, optional): Whether to return the PIL images instead of writing them to disk. Defaults to False.
        pages (Optional[List[int]], optional): The pages to rasterize. If None, all pages will be rasterized. Defaults to None.

    Returns:
        Optional[List[io.BytesIO]]: The PIL images if `return_pil` is True, otherwise None.
    """
    pils = []
    if outpath is None:
        return_pil = True
    try:
        pdf = pypdfium2.PdfDocument(pdf)
        if pages is None:
            pages = range(len(pdf))
        renderer = pdf.render(
            pypdfium2.PdfBitmap.to_pil,
            page_indices=pages,
            scale=dpi / 72,
        )
        for i, image in zip(pages, renderer):
            if return_pil:
                page_bytes = io.BytesIO()
                image.save(page_bytes, 'bmp')
                pils.append(page_bytes)
            else:
                image.save((outpath / ('%02d.png' % (i + 1))), 'png')
    except Exception as e:
        logging.error(e)
    if return_pil:
        return pils


class PDFDataset(LazyDataset):
    """
    Lazy loading dataset for processing PDF documents.

    This dataset allows lazy loading of PDF documents and provides access to processed images
    using a specified preparation function.

    Args:
        pdf (str): Path to the PDF document.
        prepare (Callable): A preparation function to process the images.

    Attributes:
        name (str): Name of the PDF document.
    """

    def __init__(self, pdf, prepare: Callable, name: Optional[str] = None, pages: Optional[List[int]] = None):
        self.prepare = prepare
        if name is not None:
            self.name = name
        elif isinstance(pdf, str):
            self.name = str(pdf)
        else:
            self.name = str(uuid.uuid4())
        self.init_fn = partial(rasterize_paper, pdf, pages=pages)
        self.dataset = None
        self.size = len(pypdfium2.PdfDocument(pdf)) if pages is None else len(pages)


def load_pdf_with_nougat(
        pdfs: List[Tuple[str, ByteString]],
        model_name: str = '0.1.0-small',
        full_precision: bool = False,
        batch_size: int = 4,
        use_cuda: bool = True,
        use_markdown: bool = True,
    ) -> List[str]:
    """Converts PDFs to text using the Meta's Nougat model.
    
    Args:
        pdfs: List of tuples containing the name and byte string contents of the PDFs to convert.
        model_name: The name of the Nougat model to use: {0.1.0-small | 0.1.0-base}.
            Model sizes: 0.1.0-small - 250M params, 0.1.0-base - 350M params.
            Note that in the paper, both models where reported to have nearly identical performance.
        full_precision: Use fp32 when True, bf16 when False.
        batch_size: The batch size to use.
        use_cuda: Whether to use CUDA or not (will automatically use cpu if cuda not available).
        use_markdown: Output text is converted to markdown when True.
        
    Returns:
        A list of strings containing the text from the PDFs.
    """
    checkpoint = get_checkpoint(model_tag=model_name)
    model = NougatModel.from_pretrained(checkpoint)
    model = move_to_device(model, bf16=not full_precision, cuda=use_cuda)

    model.eval()
    datasets = []
    for (name, pdf) in pdfs:
        dataset = PDFDataset(
            pdf,
            partial(model.encoder.prepare_input, random_padding=False),
            name = name,
        )
        datasets.append(dataset)

    if len(datasets) == 0:
        return []

    dataloader = torch.utils.data.DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PDFDataset.ignore_none_collate,
    )

    predictions = []
    file_index = 0
    page_num = 0
    pdf_contents = []

    for i, (sample, is_last_page) in enumerate(tqdm(dataloader)):

        model_output = model.inference(
            image_tensors=sample, early_stopping=True)
        
        # Check if model output is faulty
        for j, output in enumerate(model_output['predictions']):
            if page_num == 0:
                logging.info(
                    "Processing file %s with %i pages"
                    % (datasets[file_index].name, datasets[file_index].size)
                )
            page_num += 1

            if output.strip() == '[MISSING_PAGE_POST]':
                # Uncaught repetitions -- most likely empty page
                predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
            elif model_output['repeats'][j] is not None:
                if model_output['repeats'][j] > 0:
                    # If we end up here, it means the output is most likely not complete and was truncated.
                    logging.warning(f"Skipping page {page_num} due to repetitions.")
                    predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                else:
                    # If we end up here, it means the document page is too different from the training domain.
                    # This can happen e.g. for cover pages.
                    predictions.append(
                        f"\n\n[MISSING_PAGE_EMPTY:{i*batch_size+j+1}]\n\n"
                    )
            else:
                if use_markdown:
                    output = markdown_compatible(output)
                predictions.append(output)

            if is_last_page[j]:
                out = ''.join(predictions).strip()
                out = re.sub(r'\n{3,}', '\n\n', out).strip()
                pdf_contents.append(out)

                predictions = []
                page_num = 0
                file_index += 1

    return pdf_contents


def load_pdf(byte_string: ByteString, library: str = 'nougat', **kwargs) -> Optional[str]:
    """Converts a PDF byte string to text using the specified library.

    Args:
        byte_string: The byte string contents of the PDF to convert.
        library: The library to use for PDF conversion: {nougat | pymupdf}.
            Nougat takes longer, but is significantly better for research papers.
    
    Returns:
        A string containing the text from the PDF.
    """
    if library == 'nougat':
        contents = load_pdf_with_nougat([(None, byte_string)], **kwargs)
        if len(contents) == 0:
            markdown = None
        else:
            markdown = contents[0]
    elif library == 'pymupdf':
        document = pymupdf.open('pdf', byte_string)
        markdown = pymupdf4llm.to_markdown(document)
    else:
        raise ValueError(f"Unsupported library: {library}")

    return markdown


def load_generic_text_doc(byte_string: ByteString, **kwargs):
    result = from_bytes(byte_string).best()
    return None if result is None else str(result)


CONVERSION_HANDLER_REGISTRY = {
    'pdf': load_pdf,
}


def retrieve_from_cache(byte_string: ByteString, cache_dir: str):
    """Retrieve text from cache using the input byte string hash as the key."""
    hash = hashlib.md5(byte_string).hexdigest()
    cache_path = Path(cache_dir) / f'{hash}.md'
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return f.read()
    return None


def save_to_cache(byte_string: ByteString, cache_dir: str, text: str):
    """Save text to cache using the input byte string hash as the key."""
    hash = hashlib.md5(byte_string).hexdigest()
    cache_path = Path(cache_dir) / hash
    os.makedirs(cache_path.parent, exist_ok=True)
    with open(cache_path, 'w') as f:
        f.write(text)


def read_from_bytes(
        byte_string: ByteString,
        file_type: str,
        cache: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> str:
    """Convert the byte string to text with file-type specific conversion handlers."""
    use_cache = cache and cache_dir

    if use_cache:
        cached = retrieve_from_cache(byte_string, cache_dir)
        if cached is not None:
            return cached

    handler_fn = CONVERSION_HANDLER_REGISTRY.get(file_type.lower(), load_generic_text_doc)
    text = handler_fn(byte_string, **kwargs)
    text = text or ''

    if use_cache:
        save_to_cache(byte_string, cache_dir, text)

    return text


if __name__ == '__main__':
    with open('dreamer_v3_paper.pdf', 'rb') as f:
        byte_string = f.read()

    with open('dreamer_v3_paper.md', 'w') as f:
        f.write(read_from_bytes(byte_string, 'pdf', batch_size=8))