document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('uploadForm').addEventListener('submit', async function(event) {
        console.log('Proessing image')
        event.preventDefault(); 

        document.getElementById('loading').style.display = 'block'; 
        document.getElementById('content').style.display = 'none'; 

        const formData = new FormData(this); 
        const response = await fetch('/process/', {
            method: 'POST',
            body: formData,
        });
        const htmlContent = await response.text(); 

        document.getElementById('loading').style.display = 'none'; 

        document.getElementById('content').innerHTML = htmlContent;
        document.getElementById('content').style.display = 'block'; 
    });

    document.getElementById('imagePath').addEventListener('change', function(event) {
        document.getElementById('imagePathTextBox').value = this.value;
    });
})