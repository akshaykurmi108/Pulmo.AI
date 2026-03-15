/* =========================
   EMAIL VALIDATION
========================= */

function validateEmail(){

let email = document.getElementById("email").value.trim();

let pattern = /^[^ ]+@[^ ]+\.[a-z]{2,3}$/;

if(email === ""){
alert("Please enter your email");
return false;
}

if(!email.match(pattern)){
alert("Enter a valid email address");
return false;
}

sendOTP();

return false;

}


/* =========================
   SEND OTP (SIMULATION)
========================= */

function sendOTP(){

let otp = Math.floor(100000 + Math.random()*900000);

localStorage.setItem("generatedOTP", otp);

alert("OTP sent to your email: " + otp);

document.getElementById("otpSection").style.display="block";

}


/* =========================
   VERIFY OTP
========================= */

function verifyOTP(){

let enteredOTP = document.getElementById("otp").value;

let savedOTP = localStorage.getItem("generatedOTP");

if(enteredOTP === savedOTP){

alert("OTP Verified Successfully");

document.getElementById("resetSection").style.display="block";

}
else{

alert("Invalid OTP");

}

}


/* =========================
   RESET PASSWORD
========================= */

function resetPassword(){

let pass1 = document.getElementById("newPassword").value;
let pass2 = document.getElementById("confirmPassword").value;

if(pass1.length < 6){

alert("Password must be at least 6 characters");

return;

}

if(pass1 !== pass2){

alert("Passwords do not match");

return;

}

alert("Password Reset Successful");

window.location.href="login.html";

}


/* =========================
   INPUT HIGHLIGHT EFFECT
========================= */

let inputs = document.querySelectorAll("input");

inputs.forEach(input => {

input.addEventListener("focus", function(){

input.style.borderColor="#1f6d95";
input.style.boxShadow="0 0 5px rgba(31,109,149,0.3)";

});

input.addEventListener("blur", function(){

input.style.borderColor="#ccc";
input.style.boxShadow="none";

});

});