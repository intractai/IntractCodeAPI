/* shorthand functions (createElement is defined at bottom)*/
const div = (props, ...children) => createElement("div", props, ...children);
const ul = (props, ...children) => createElement("ul", props, ...children);
const li = (props, ...children) => createElement("li", props, ...children);
const i = (props, ...children) => createElement("i", props, ...children);
const span = (props, ...children) => createElement("span", props, ...children);
const header = (props, ...children) =>
  createElement("header", props, ...children);
const p = (props, ...children) => createElement("p", props, ...children);
const section = (props, ...children) =>
  createElement("section", props, ...children);
const button = (props, ...children) =>
  createElement("button", props, ...children);
const input = (props, ...children) => createElement("input", props, ...children);


// Modified input function to include label creation for checkboxes
function checkboxWithLabel(props, ...children) {
    // Create a wrapper for the checkbox
    const wrapper = createElement('label', { className: 'custom-checkbox-wrapper' });
    const checkbox = createElement("input", props, ...children);
    const checkmark = createElement('span', { className: 'checkmark' });
    
    wrapper.appendChild(checkbox);
    wrapper.appendChild(checkmark);
    
    return wrapper;
}


/* File */

const File = ({ name }) => {
    return div(
        { className: "file" },
        input({ type: "checkbox", className: "item-checkbox", onChange: toggleSelection }), // Add checkbox
        i({ className: "material-icons", style: "opacity: 0;" }, "arrow_right"),
        i({ className: "material-icons" }, "insert_drive_file"),
        span(null, name)
    );
  };


/* Folder */

const Folder = (props, ...children) => {
    const opened = props.opened || false;
    const arrowIcon = opened ? openedArrowIcon : closedArrowIcon;
    const folderIcon = opened ? openedFolderIcon : closedFolderIcon;
    const folderName = props.name || "unknown";

    return div(
        { className: "folder" },
        header(
        // Attach the event listener here, to the entire header
        { className: "folder-header", opened: opened, onClick: changeOpened },
        checkboxWithLabel({ type: "checkbox", className: "item-checkbox", onChange: toggleSelection }), // Keep checkbox functionality
        i({ className: "material-icons" }, arrowIcon),
        i({ className: "material-icons" }, folderIcon),
        span(null, folderName)
        ),
        ul({ className: opened ? "" : "hide" }, ...children)
    );
};

const openedFolderIcon = "folder_open";
const closedFolderIcon = "folder";
const openedArrowIcon = "arrow_drop_down";
const closedArrowIcon = "arrow_right";

function changeOpened(event) {
    // Skip if the click is on the checkbox
    if (event.target.type === 'checkbox') return; 

    event.stopPropagation(); // Prevent further propagation of the event
    const folderHeader = event.currentTarget; // Use currentTarget for the element the event was actually attached to
    const opened = folderHeader.getAttribute("opened") === "true";
    const newOpened = !opened;

    // Use the specific classes for selection
    const arrowIcon = folderHeader.querySelector(".arrow-icon");
    const folderIcon = folderHeader.querySelector(".folder-icon");

    if (arrowIcon && folderIcon) { // Check if elements are found to prevent errors
        arrowIcon.textContent = newOpened ? openedArrowIcon : closedArrowIcon;
        folderIcon.textContent = newOpened ? openedFolderIcon : closedFolderIcon;
    } else {
        // Log an error or handle the case where icons are not found
        console.error("Icons not found");
    }

    // Toggle content visibility
    const content = folderHeader.nextElementSibling;
    if (content) { // Check if content is found
        content.classList.toggle("hide", !newOpened);
    } else {
        console.error("Content element not found");
    }

    folderHeader.setAttribute("opened", newOpened.toString());
}

/* My react-clone mini library */

function appendChildren(parent, children) {
  for (let child of children) {
    if (!child) continue;
    switch (typeof child) {
      case "string":
        const el = document.createTextNode(child);
        parent.appendChild(el);
        break;
      default:
        parent.appendChild(child);
        break;
    }
  }
}
function setStyle(el, style) {
  if (typeof style == "string") {
    el.setAttribute("style", style);
  } else {
    Object.assign(el.style, style);
  }
}
function setClass(el, className) {
  className.split(/\s/).forEach(element => {
    if (element) {
      el.classList.add(element);
    }
  });
}
function setProps(el, props) {
  const eventRegex = /^on([a-z]+)$/i;
  for (let propName in props) {
    if (!propName) continue;

    if (propName === "style") {
      setStyle(el, props[propName]);
    } else if (propName === "className") {
      setClass(el, props[propName]);
    } else if (eventRegex.test(propName)) {
      const eventToListen = propName.replace(eventRegex, "$1").toLowerCase();
      el.addEventListener(eventToListen, props[propName]);
    } else {
      el.setAttribute(propName, props[propName]);
    }
  }
}

//type, [props], [...children] 
function createElement(type, props, ...children) {
  if (typeof type === "function") {
    return type(props);
  } else {
    const el = document.createElement(type);
    if (props && typeof props === "object") {
      setProps(el, props);
    }
    if (children) {
      appendChildren(el, children);
    }
    return el;
  }
}


/* On file upload */

function createComponentsFromStructure(structure, name = '') {
    // Base case: If the structure is a file (null), return an array with a File component
    if (structure === null) {
        return [File({ name })];
    }

    // Recursive case: For folders, map through each key to create components, ensuring each call returns an array
    const children = Object.keys(structure).flatMap(key => 
        createComponentsFromStructure(structure[key], key)
    );

    // Return an array with a single Folder component that contains all children
    return [Folder({ name }, ...children)];
}

document.getElementById('directory-picker').addEventListener('change', function(event) {
    const files = event.target.files;
    // const fileTree = document.createElement('ul');
    const fileStructure = {};
  
    for (let file of files) {
      const path = file.webkitRelativePath.split('/');
      let currentLevel = fileStructure;
  
      path.forEach((part, index) => {
        if (!currentLevel[part]) {
          currentLevel[part] = index === path.length - 1 ? null : {};
        }
        currentLevel = currentLevel[part];
      });
    }

    // Next we need to convert the file structure into a `section`
    // of files and folders.


    const TreeView = () => {
        // Convert the fileStructure into components
        const fileComponents = createComponentsFromStructure(fileStructure, 'Project Directory');
        return section({ className: "container" }, ...fileComponents);
      };


    const app = document.querySelector("#treeView");
    // Clear existing content before appending new tree view
    while (app.firstChild) {
      app.removeChild(app.firstChild);
    }
    app.appendChild(createElement(TreeView));
});


/* Handle selection */

function toggleSelection(event) {
    const checkbox = event.target;
    const parentElement = checkbox.closest('.folder, .file');
    
    const toggleRecursive = (element, isChecked) => {
      const checkboxes = element.querySelectorAll('.item-checkbox');
      checkboxes.forEach(chk => chk.checked = isChecked);
    };
  
    // Check if the parent element is a folder for recursive toggle
    if (parentElement.classList.contains('folder')) {
      toggleRecursive(parentElement, checkbox.checked);
    }
}