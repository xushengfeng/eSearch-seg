const x = require("../");

start();

async function start() {
    await x.init({
        segPath: "./m/398x224softmax.onnx",
    });
    let cameraStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: true,
    });
    v.srcObject = cameraStream;
    v.play();
}
let v = document.createElement("video");
let i = document.createElement("canvas");
document.body.append(i);
let out = document.createElement("canvas");
document.body.append(out);

function draw() {
    const canvasCtx = i.getContext("2d");
    i.width = v.videoWidth;
    i.height = v.videoHeight;
    canvasCtx.drawImage(v, 0, 0, i.width, i.height);
    x.seg(i.getContext("2d").getImageData(0, 0, i.width, i.height)).then((data) => {
        out.width = data.width;
        out.height = data.height;
        out.getContext("2d").putImageData(data, 0, 0);
    });
    setTimeout(() => {
        draw();
    }, 10);
}
v.onloadedmetadata = draw;
