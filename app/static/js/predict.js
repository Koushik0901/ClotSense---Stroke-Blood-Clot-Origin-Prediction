const form = document.querySelector("form"),
	fileInput = document.querySelector(".file-input"),
	progressArea = document.querySelector(".progress-area"),
	uploadedArea = document.querySelector(".uploaded-area");

const result = document.getElementById("result-text");

form.addEventListener("click", () => {
	fileInput.click();
});

fileInput.onchange = ({ target }) => {
	let file = target.files[0];
	if (file) {
		let fileName = file.name;
		if (fileName.length >= 12) {
			let splitName = fileName.split(".");
			fileName = splitName[0].substring(0, 13) + "... ." + splitName[1];
		}
		uploadFile(fileName);
		result.innerHTML = `<div class="spinner-border" role="status">
  <span class="visually-hidden">Loading...</span>
</div>`;
	}
};

function uploadFile(name) {
	let xhr = new XMLHttpRequest();
	xhr.open("POST", "/predict");
	xhr.onload = () => {
		var prediction = JSON.parse(xhr.responseText);
		result.innerHTML = prediction["class"] + " is the predicted class";
	};
	let data = new FormData(form);
	xhr.send(data);
}
