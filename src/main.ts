var cv = require("opencv.js");
var ort: typeof import("onnxruntime-web");

export { x as seg, init };

var dev = true;
type AsyncType<T> = T extends Promise<infer U> ? U : never;
type SessionType = AsyncType<ReturnType<typeof import("onnxruntime-web").InferenceSession.create>>;
var det: SessionType;

async function init(x: { segPath: string; node?: boolean; dev?: boolean; ort?: typeof import("onnxruntime-web") }) {
    if (x.ort) {
        ort = x.ort;
    } else {
        if (x.node) {
            ort = require("onnxruntime-node");
        } else {
            ort = require("onnxruntime-web");
        }
    }
    dev = x.dev;
    det = await ort.InferenceSession.create(x.segPath);
    return new Promise((rs) => rs(true));
}

/** 主要操作 */
async function x(img: ImageData) {
    if (dev) console.time();
    let { transposedData, image } = beforeSeg(img);
    const detResults = await runSeg(transposedData, image, det);
    if (dev) {
        console.log(detResults);
        console.timeEnd();
    }

    let data = afterSeg(detResults.data, detResults.dims[3], detResults.dims[2], img);
    return data;
}

async function runSeg(transposedData: number[][][], image: ImageData, det: SessionType) {
    let x = transposedData.flat(Infinity) as number[];
    const detData = Float32Array.from(x);

    const detTensor = new ort.Tensor("float32", detData, [1, 3, image.height, image.width]);
    let detFeed = {};
    detFeed[det.inputNames[0]] = detTensor;

    const detResults = await det.run(detFeed);
    return detResults[det.outputNames[0]];
}

function data2canvas(data: ImageData, w?: number, h?: number) {
    let x = document.createElement("canvas");
    x.width = w || data.width;
    x.height = h || data.height;
    x.getContext("2d").putImageData(data, 0, 0);
    return x;
}

/**
 *
 * @param {ImageData} data 原图
 * @param {number} w 输出宽
 * @param {number} h 输出高
 */
function resizeImg(data: ImageData, w: number, h: number) {
    let x = data2canvas(data);
    let src = document.createElement("canvas");
    src.width = w;
    src.height = h;
    src.getContext("2d").scale(w / data.width, h / data.height);
    src.getContext("2d").drawImage(x, 0, 0);
    return src.getContext("2d").getImageData(0, 0, w, h);
}

function beforeSeg(image: ImageData) {
    // image = resizeImg(image, 256, 144);
    image = resizeImg(image, 398, 224);

    const transposedData = toPaddleInput(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
    if (dev) {
        let srcCanvas = data2canvas(image);
        document.body.append(srcCanvas);
    }
    return { transposedData, image };
}

function afterSeg(data: AsyncType<ReturnType<typeof runSeg>>["data"], w: number, h: number, srcData: ImageData) {
    var myImageData = new ImageData(w, h);
    for (let i = 0; i < w * h; i++) {
        let n = Number(i) * 4;
        const v = (data[i] as number) > 0.8 ? 0 : 255;
        myImageData.data[n] = myImageData.data[n + 1] = myImageData.data[n + 2] = myImageData.data[n + 3] = v;
    }
    let mask = resizeImg(myImageData, srcData.width, srcData.height);
    if (dev) {
        let x = data2canvas(mask);
        document.body.append(x);
    }
    let resultImageData = new ImageData(srcData.width, srcData.height);
    for (let i = 0; i < mask.data.length; i++) {
        resultImageData.data[i] = srcData.data[i] * (mask.data[i] / 255);
    }
    if (dev) {
        let x = data2canvas(resultImageData);
        document.body.append(x);
    }
    return resultImageData;
}

function toPaddleInput(image: ImageData, mean: number[], std: number[]) {
    const imagedata = image.data;
    const redArray: number[][] = [];
    const greenArray: number[][] = [];
    const blueArray: number[][] = [];
    let x = 0,
        y = 0;
    for (let i = 0; i < imagedata.length; i += 4) {
        if (!blueArray[y]) blueArray[y] = [];
        if (!greenArray[y]) greenArray[y] = [];
        if (!redArray[y]) redArray[y] = [];
        redArray[y][x] = (imagedata[i] / 255 - mean[0]) / std[0];
        greenArray[y][x] = (imagedata[i + 1] / 255 - mean[1]) / std[1];
        blueArray[y][x] = (imagedata[i + 2] / 255 - mean[2]) / std[2];
        x++;
        if (x == image.width) {
            x = 0;
            y++;
        }
    }

    return [blueArray, greenArray, redArray];
}
