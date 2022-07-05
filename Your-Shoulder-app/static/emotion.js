let personnet;
const webcam2 = new Webcam(document.getElementById('wc'));
let isPredicting2 = false;
var personPrediction = 'Neutral';

async function loadPersonnet() {
    const personnet = await tf.loadLayersModel('static/models/emotion_cnn/model.json');
    console.log("Model loaded");
    return personnet;
}

// function cnnResponse(img1) {
// 	// Video Response
// 	console.log("Hy there I get called!! lets see about python");

// 	$.get("/res", { img: img1 }).done(function () {
// 	//   const msgImg = data;
// 		console.log('img');
// 	});

//   }

async function predict2() {
    console.log("predict emotion method");
  while (isPredicting2) {
    const predictedClass = tf.tidy(() => {
      const img = webcam2.capture();

	  // call python face detection
	//   cnnResponse(img);
		// console.log(img)

      const prediction = personnet.predict(img);
      return prediction.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText2 = "";
    switch(classId){
		case 0:
			predictionText2 = "Angry";
			personPrediction = predictionText2;
			break;
		case 1:
			predictionText2 = "Sad";
			personPrediction = predictionText2;
//			window.location.replace("complaint/id=".concat("stud1"))
			break;
        case 2:
			predictionText2 = "Neutral";
			personPrediction = predictionText2;
			break;
        case 3:
			predictionText2 = "Happy";
			personPrediction = predictionText2;
			break;
	}
	document.getElementById("prediction2").innerText = predictionText2;
    predictedClass.dispose();
    await tf.nextFrame();
  }
}

function startPredicting(){
	isPredicting2 = true;
	predict2();
}

function stopPredicting(){
	isPredicting2 = false;
	predict2();
}

async function init(){
	await webcam2.setup();
	personnet = await loadPersonnet();
	tf.tidy(() => personnet.predict(webcam2.capture()));	
}

init();