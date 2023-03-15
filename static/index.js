const wrapper = document.querySelector(".wrapper");
const loginlink = document.querySelector(".login-link");
const registerlink = document.querySelector(".register-link");

const btnpopup = document.querySelector(".btnLogin-popup");
const iconclose = document.querySelector(".icon-close");

registerlink.addEventListener("click", () => {
	wrapper.classList.add("active");
});

loginlink.addEventListener("click", () => {
	wrapper.classList.remove("active");
});

btnpopup.addEventListener("click", () => {
	wrapper.classList.add("active-popup");
});

iconclose.addEventListener("click", () => {
	wrapper.classList.remove("active-popup");
});
email.addEventListener("textinput", email_verify);
password.addEventListener("textinput", pass_verify);

var email = document.forms["form"]["mail"];
var password = document.forms["form"]["password"];

function validated() {
	if (email.value.length < 9) {
		email.focus();
		return false;
	}
	if (password.value.length < 5) {
		password.focus();
		return false;
	}
}

function email_verify() {
	if (email.value.length >= 8) {
		return true;
	}
}
function pass_verify() {
	if (password.value.length >= 8) {
		return true;
	}
}