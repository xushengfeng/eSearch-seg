# eSearch-seg

本仓库是 [eSearch](https://github.com/xushengfeng/eSearch)的录屏人像识别依赖

基于 [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)

基于[onnxruntime](https://github.com/microsoft/onnxruntime)的 web runtime，使用 wasm 运行，未来可能使用 webgl 甚至是 webgpu。

模型需要转换为 onnx 才能使用：[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) 或[在线转换](https://www.paddlepaddle.org.cn/paddle/visualdl/modelconverter/x2paddle)

在 js 文件下使用 electron 进行调试（需要 HTML canvas，暂不支持纯 nodejs）

## 使用

```shell
npm i esearch-seg
```

web

```javascript
import * as seg from "esearch-seg";
```

nodejs

```javascript
const seg = require("esearch-seg");
```

```javascript
await seg.init({
    segPath: "seg.onnx",
});

let img = document.createElement("img");
img.src = "data:image/png;base64,...";
img.onload = async () => {
    let canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.getContext("2d").drawImage(img, 0, 0);
    seg.seg(canvas.getContext("2d").getImageData(0, 0, img.width, img.height))
        .then((data) => {})
        .catch((e) => {});
};
```

init type

```typescript
{
    segPath: string;
    node?: boolean;
    dev?: boolean;
    ort?: typeof import("onnxruntime-web");
```

ocr type

```typescript
seg(img: ImageData): Promise<ImageData> // 人像
```
