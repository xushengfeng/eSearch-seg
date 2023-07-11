const x = require("../");
const fs = require("fs");

start();

async function start() {
    await x.init({
        detPath: "./m/ch_PP-OCRv3_det_infer.onnx",
        recPath: "./m/jp/japan_rec.onnx",
        dic: fs.readFileSync("./m/jp/japan_dict.txt").toString(),
        dev: true,
        node: true,
    });
    let img = document.createElement("img");
    img.src = "../a8.png";
    img.onload = () => {
        let canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        canvas.getContext("2d").drawImage(img, 0, 0);
        x.ocr(canvas.getContext("2d").getImageData(0, 0, img.width, img.height)).then((v) => {
            let tl = [];
            for (let i of v) {
                tl.push(i.text);
            }
            let p = document.createElement("p");
            p.innerText = tl.join("\n");
            document.body.append(p);
        });
    };
}
