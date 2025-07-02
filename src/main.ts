var ort: typeof import("onnxruntime-node");

export { x as seg, init };

var dev = true;
type AsyncType<T> = T extends Promise<infer U> ? U : never;
type SessionType = AsyncType<
    ReturnType<typeof import("onnxruntime-node").InferenceSession.create>
>;
var det: SessionType;
var shape = [512, 512];
var invertOpacity = false;
var threshold = 0;

async function init(x: {
    segPath: string;
    dev?: boolean;
    ort: typeof import("onnxruntime-node");
    ortOption?: import("onnxruntime-node").InferenceSession.SessionOptions;
    shape: [number, number];
    invertOpacity?: boolean;
    threshold?: number;
}) {
    ort = x.ort;
    dev = x.dev;
    det = await ort.InferenceSession.create(x.segPath, x.ortOption);
    if (x.shape) shape = x.shape;
    if (x.invertOpacity) invertOpacity = x.invertOpacity;
    if (x.threshold) threshold = x.threshold;
    return new Promise((rs) => rs(true));
}

/** 主要操作 */
async function x(img: ImageData) {
    if (dev) console.time();
    const { transposedData } = beforeSeg(img);
    const detResults = await runSeg(transposedData, det);
    if (dev) {
        console.log(detResults);
        console.timeEnd();
    }

    const data = afterSeg(
        detResults.data,
        detResults.dims[3],
        detResults.dims[2],
        img,
    );
    return data;
}

async function runSeg(transposedData: number[][][], det: SessionType) {
    const x = transposedData.flat(2) as number[];
    const detData = Float32Array.from(x);

    const detTensor = new ort.Tensor("float32", detData, [
        1,
        3,
        transposedData[0].length,
        transposedData[0][0].length,
    ]);
    const detFeed = {};
    detFeed[det.inputNames[0]] = detTensor;

    const detResults = await det.run(detFeed);
    return detResults[det.outputNames[0]];
}

function data2canvas(data: ImageData, w?: number, h?: number) {
    const x = document.createElement("canvas");
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
    const x = data2canvas(data);
    const src = document.createElement("canvas");
    src.width = w;
    src.height = h;
    src.getContext("2d").scale(w / data.width, h / data.height);
    src.getContext("2d").drawImage(x, 0, 0);
    return src.getContext("2d").getImageData(0, 0, w, h);
}

function beforeSeg(image: ImageData) {
    image = resizeImg(image, shape[0], shape[1]);

    const transposedData = toPaddleInput(
        image,
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    );
    if (dev) {
        const srcCanvas = data2canvas(image);
        document.body.append(srcCanvas);
    }
    return { transposedData, image };
}

function afterSeg(
    data: AsyncType<ReturnType<typeof runSeg>>["data"],
    w: number,
    h: number,
    srcData: ImageData,
) {
    const myImageData = new ImageData(w, h);
    for (let i = 0; i < w * h; i++) {
        const n = Number(i) * 4;
        const v = 255 * (data[i] as number);
        myImageData.data[n] =
            myImageData.data[n + 1] =
            myImageData.data[n + 2] =
                0;
        myImageData.data[n + 3] = invertOpacity ? 255 - v : v;
    }
    const maskEl = data2canvas(myImageData);
    if (dev) {
        document.body.append(maskEl);
    }

    const newMaskData = maskEl
        .getContext("2d")
        .getImageData(0, 0, maskEl.width, maskEl.height);
    const mask = resizeImg(newMaskData, srcData.width, srcData.height);
    for (let i = 0; i < mask.data.length; i += 4) {
        const op = mask.data[i + 3] < threshold * 255 ? 0 : mask.data[i + 3];
        srcData.data[i + 3] = op;
        if (op === 0) {
            srcData.data[i] = 0;
            srcData.data[i + 1] = 0;
            srcData.data[i + 2] = 0;
        }
    }
    if (dev) {
        const x = data2canvas(srcData);
        document.body.append(x);
    }
    return srcData;
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
        if (x === image.width) {
            x = 0;
            y++;
        }
    }

    return [blueArray, greenArray, redArray];
}
