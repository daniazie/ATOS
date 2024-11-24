// var button = document.getElementById("play");
// var audio = document.getElementById("player");

let button = document.querySelectorAll("#play");
let audio = document.querySelectorAll("#player");

for (let i = 0; i < button.length; i++) {
    button[i].addEventListener("click", function() {
            if(audio[i].paused){
                audio[i].play();
                button[i].className = "fa-regular fa-circle-pause";
            } else{
                audio[i].pause();
                button[i].className = "fa-regular fa-circle-play";
            }  
    });
}