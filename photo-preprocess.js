
/*
 Bengali Handwriting Recognition
 Improved Photo Preprocessing

 Input:
 One dark handwritten Bengali character
 on light-colored paper.

 Output:
 64x64 grayscale image
 White handwriting on black background.
*/

window.BanglaPhoto = (() => {

    const SIZE = 64;

    function clamp(value) {
        return Math.max(0, Math.min(1, value));
    }

    function process(image) {

        const originalWidth =
            image.naturalWidth || image.width;

        const originalHeight =
            image.naturalHeight || image.height;

        if (!originalWidth || !originalHeight) {
            throw new Error("Image is not loaded.");
        }

        // STEP 1: Resize photograph

        const maxSide = 800;

        const scale = Math.min(
            1,
            maxSide / Math.max(
                originalWidth,
                originalHeight
            )
        );

        const width = Math.max(
            1,
            Math.round(originalWidth * scale)
        );

        const height = Math.max(
            1,
            Math.round(originalHeight * scale)
        );

        const canvas =
            document.createElement("canvas");

        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext(
            "2d",
            { willReadFrequently: true }
        );

        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);

        ctx.imageSmoothingEnabled = true;

        ctx.drawImage(
            image,
            0, 0,
            width, height
        );

        const imageData = ctx.getImageData(
            0, 0, width, height
        );

        const rgba = imageData.data;
        const count = width * height;

        // STEP 2: Convert image to grayscale

        const gray = new Float32Array(count);

        for (let i = 0; i < count; i++) {

            const p = i * 4;

            gray[i] =
                0.299 * rgba[p] +
                0.587 * rgba[p + 1] +
                0.114 * rgba[p + 2];
        }

        // STEP 3: Estimate local background
        // using an integral image

        const stride = width + 1;

        const integral = new Float64Array(
            (width + 1) * (height + 1)
        );

        for (let y = 0; y < height; y++) {

            let rowSum = 0;

            for (let x = 0; x < width; x++) {

                rowSum += gray[y * width + x];

                integral[
                    (y + 1) * stride + x + 1
                ] =
                    integral[
                        y * stride + x + 1
                    ] + rowSum;
            }
        }

        // STEP 4: Preserve dark handwriting
        // while reducing background shadows

        const strength =
            new Float32Array(count);

        const strong =
            new Uint8Array(count);

        const radius = Math.max(
            12,
            Math.round(
                Math.min(width, height) * 0.06
            )
        );

        for (let y = 0; y < height; y++) {

            for (let x = 0; x < width; x++) {

                const x1 = Math.max(0, x - radius);
                const y1 = Math.max(0, y - radius);

                const x2 = Math.min(
                    width - 1,
                    x + radius
                );

                const y2 = Math.min(
                    height - 1,
                    y + radius
                );

                const sum =
                    integral[
                        (y2 + 1) * stride + x2 + 1
                    ] -
                    integral[
                        y1 * stride + x2 + 1
                    ] -
                    integral[
                        (y2 + 1) * stride + x1
                    ] +
                    integral[
                        y1 * stride + x1
                    ];

                const area =
                    (x2 - x1 + 1) *
                    (y2 - y1 + 1);

                const background = sum / area;

                const i = y * width + x;

                const localInk = clamp(
                    (background - gray[i] - 4) / 75
                );

                const darkInk = clamp(
                    (205 - gray[i]) / 120
                ) * 0.5;

                let ink = Math.max(
                    localInk,
                    darkInk
                );

                if (gray[i] > 245) {
                    ink = 0;
                }

                strength[i] = ink;

                // Strong pixels locate the handwriting.
                // Weak pixels are retained later for
                // smooth, natural-looking strokes.

                if (ink > 0.24) {
                    strong[i] = 1;
                }
            }
        }

        // STEP 5: Remove isolated noise.
        // Keep all significant strokes, including
        // disconnected parts of Bengali characters.

        const supported =
            new Uint8Array(count);

        const columns =
            new Int32Array(width);

        const rows =
            new Int32Array(height);

        let totalStrong = 0;

        const borderX = Math.floor(width * 0.02);
        const borderY = Math.floor(height * 0.02);

        for (
            let y = borderY;
            y < height - borderY;
            y++
        ) {

            for (
                let x = borderX;
                x < width - borderX;
                x++
            ) {

                const i = y * width + x;

                if (!strong[i]) continue;

                let neighbors = 0;

                for (
                    let dy = -1;
                    dy <= 1;
                    dy++
                ) {

                    for (
                        let dx = -1;
                        dx <= 1;
                        dx++
                    ) {

                        const nx = x + dx;
                        const ny = y + dy;

                        if (
                            nx >= 0 &&
                            nx < width &&
                            ny >= 0 &&
                            ny < height
                        ) {

                            if (
                                strong[ny * width + nx]
                            ) {
                                neighbors++;
                            }
                        }
                    }
                }

                if (neighbors >= 2) {

                    supported[i] = 1;

                    columns[x]++;
                    rows[y]++;

                    totalStrong++;
                }
            }
        }

        if (totalStrong < 15) {

            throw new Error(
                "No clear handwriting detected. " +
                "Please use a darker pen or " +
                "a clearer photograph."
            );
        }

        if (totalStrong > count * 0.35) {

            throw new Error(
                "Too much dark background detected. " +
                "Please crop the image around " +
                "one character."
            );
        }

        // STEP 6: Find the character bounds.
        // Trim a very small amount of outlying
        // noise without cutting off the letters.

        function findLowerBound(histogram, trim) {

            let seen = 0;

            for (
                let i = 0;
                i < histogram.length;
                i++
            ) {

                seen += histogram[i];

                if (seen > trim) {
                    return i;
                }
            }

            return 0;
        }

        function findUpperBound(histogram, trim) {

            let seen = 0;

            for (
                let i = histogram.length - 1;
                i >= 0;
                i--
            ) {

                seen += histogram[i];

                if (seen > trim) {
                    return i;
                }
            }

            return histogram.length - 1;
        }

        const trim =
            Math.floor(totalStrong * 0.003);

        let minX = findLowerBound(
            columns, trim
        );

        let maxX = findUpperBound(
            columns, trim
        );

        let minY = findLowerBound(
            rows, trim
        );

        let maxY = findUpperBound(
            rows, trim
        );

        if (maxX <= minX || maxY <= minY) {

            throw new Error(
                "Could not identify character bounds."
            );
        }

        // Add a small margin around the strokes

        const padding = Math.max(
            2,
            Math.round(
                Math.max(
                    maxX - minX,
                    maxY - minY
                ) * 0.035
            )
        );

        minX = Math.max(0, minX - padding);
        minY = Math.max(0, minY - padding);

        maxX = Math.min(
            width - 1,
            maxX + padding
        );

        maxY = Math.min(
            height - 1,
            maxY + padding
        );

        // STEP 7: Create a smooth grayscale mask

        const cleanCanvas =
            document.createElement("canvas");

        cleanCanvas.width = width;
        cleanCanvas.height = height;

        const cleanCtx =
            cleanCanvas.getContext("2d");

        const cleanData =
            cleanCtx.createImageData(width, height);

        for (let y = 0; y < height; y++) {

            for (let x = 0; x < width; x++) {

                const i = y * width + x;

                let keep = supported[i] === 1;

                // Preserve softer edge pixels near
                // the detected handwriting strokes.

                if (
                    !keep &&
                    strength[i] > 0.12
                ) {

                    for (
                        let dy = -1;
                        dy <= 1 && !keep;
                        dy++
                    ) {

                        for (
                            let dx = -1;
                            dx <= 1;
                            dx++
                        ) {

                            const nx = x + dx;
                            const ny = y + dy;

                            if (
                                nx >= 0 &&
                                nx < width &&
                                ny >= 0 &&
                                ny < height &&
                                supported[
                                    ny * width + nx
                                ]
                            ) {
                                keep = true;
                                break;
                            }
                        }
                    }
                }

                const value = keep
                    ? Math.round(
                        clamp(strength[i]) * 255
                    )
                    : 0;

                const p = i * 4;

                cleanData.data[p] = value;
                cleanData.data[p + 1] = value;
                cleanData.data[p + 2] = value;
                cleanData.data[p + 3] = 255;
            }
        }

        cleanCtx.putImageData(
            cleanData,
            0,
            0
        );

        // STEP 8: Resize and center to 64x64

        const outputCanvas =
            document.createElement("canvas");

        outputCanvas.width = SIZE;
        outputCanvas.height = SIZE;

        const outputCtx =
            outputCanvas.getContext("2d");

        outputCtx.fillStyle = "black";

        outputCtx.fillRect(
            0, 0, SIZE, SIZE
        );

        const cropWidth = maxX - minX + 1;
        const cropHeight = maxY - minY + 1;

        const resizeScale = Math.min(
            52 / cropWidth,
            52 / cropHeight
        );

        const newWidth =
            cropWidth * resizeScale;

        const newHeight =
            cropHeight * resizeScale;

        outputCtx.imageSmoothingEnabled = true;
        outputCtx.imageSmoothingQuality = "high";

        outputCtx.drawImage(
            cleanCanvas,
            minX,
            minY,
            cropWidth,
            cropHeight,
            (SIZE - newWidth) / 2,
            (SIZE - newHeight) / 2,
            newWidth,
            newHeight
        );

        // STEP 9: Prepare normalized CNN input

        const outputData =
            outputCtx.getImageData(
                0, 0, SIZE, SIZE
            ).data;

        const pixels =
            new Float32Array(SIZE * SIZE);

        for (let i = 0; i < pixels.length; i++) {

            pixels[i] =
                outputData[i * 4] / 255;
        }

        return {
            pixels: pixels,
            preview: outputCanvas.toDataURL(
                "image/png"
            )
        };
    }

    return {
        process
    };

})();
