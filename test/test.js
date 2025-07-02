const x = require("../");
const ort = require("onnxruntime-node");

start();

async function start() {
    await x.init({
        segPath: "./m/398x224softmax.onnx",
        dev: true,
        node: true,
        ort,
        invertOpacity: true,
        shape: [256, 144],
    });
    const img = document.createElement("img");
    img.src = "../x.webp";
    img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        canvas.getContext("2d").drawImage(img, 0, 0);
        x.seg(
            canvas.getContext("2d").getImageData(0, 0, img.width, img.height),
        ).then((data) => {
            xx.width = data.width;
            xx.height = data.height;
            xx.getContext("2d").putImageData(data, 0, 0);
        });
    };
}
const xx = document.createElement("canvas");
document.body.append(xx);
