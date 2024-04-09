document.getElementById('add-language').addEventListener('click', function() {
    const selectedLanguage = document.getElementById('language-dropdown').value;
    const box = document.getElementById('libraries-box');
    const langDiv = document.createElement('div');
    langDiv.textContent = selectedLanguage;
    const addButton = document.createElement('button');
    addButton.textContent = '+';
    addButton.onclick = function() {
        const libInput = prompt('Enter Library Name:');
        if (libInput) {
            const libDiv = document.createElement('div');
            libDiv.textContent = libInput;
            langDiv.appendChild(libDiv);
        }
    };
    langDiv.appendChild(addButton);
    box.appendChild(langDiv);
});

document.getElementById('confirm-btn').addEventListener('click', function() {
    const data = {
        files: [], // You'll need to handle file selection and reading separately
        languages: []
    };
    
    // Dummy data submission
    console.log('Submitting:', data);
    // Here you would use fetch or another method to make the REST API POST request
});