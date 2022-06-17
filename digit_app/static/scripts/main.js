
const canvas = document.querySelector("canvas");
const context = canvas.getContext("2d");
canvas.height = 340;
canvas.width = 300;
context.fillStyle = "white";
context.fillRect(0,0, canvas.width, canvas.height)

let is_drawing = false;
let drawn_something = false;

function startPosition(event) {
    is_drawing = true;
    draw(event);
}
function finishPosition(event) {
    drawn_something = true;
    is_drawing = false;
    context.beginPath();
}
function draw(event){
    if (!is_drawing) {
        return;
    }
    context.lineWidth = parseInt(document.getElementById("ln_width").value);  

    context.lineCap = "round";
    context.lineTo(event.clientX - canvas.offsetLeft, event.clientY- canvas.offsetTop);
    context.stroke();
    context.beginPath();
    context.moveTo(event.clientX - canvas.offsetLeft,event.clientY - canvas.offsetTop);
}

function clearCanvas() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById("output_result").innerHTML = "";
    context.fillRect(0,0, canvas.width, canvas.height)  
    drawn_something = false;
}

function setOutputResult(result, prob) {
  document.getElementById("output_result").innerHTML = result + " (Confidence: " + prob + " %)"
}


function predictDigit(){
    if (!drawn_something) {
      showAlert("Please, draw a digit","warning",2);
      return;
    }

    // get the model 
    let models = document.getElementById("models");
    let selected_model = models.options[models.selectedIndex].value;
    if (selected_model == "none"){
      showAlert("Please, select a model", "warning", 2);
      return;
    }

    // convert canvas into image
    let canvas_image = new Image();
    canvas_image.src = canvas.toDataURL();
    let image_data = {
      image : canvas_image.src,
      model : selected_model,
    }

    fetch(`${window.origin}/recognize_digit/get_info`, {
      method: "POST",
      credentials: "include",
      body : JSON.stringify(image_data),
      cache : "no-cache",
      headers : new Headers({
        "content-type" : "application/json"
      })
    })
    .then(function(response){
      if (response.status != 200){
        console.log("Response states was not 200: ", response.status);
        return;
      } else {
        response.json().then(function(data){
          console.log("Message from Server: ", data["server_message"]);
          console.log("Predicted result:", data["prediction"]);
          console.log("Confidence:", data["probability"]);

          setOutputResult(data["prediction"], data["probability"]);
        })
      }
    })
}

canvas.addEventListener("mousedown", startPosition);
canvas.addEventListener("mouseup", finishPosition);
canvas.addEventListener("mousemove", draw);

