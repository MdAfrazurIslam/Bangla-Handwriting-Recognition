
/*
 Bengali Handwriting Recognition
 Photo Preprocessing

 Input: Photograph containing one dark
        handwritten character on light paper.

 Output: 64x64 grayscale pixels
         White character on black background.
*/

window.BanglaPhoto = (() => {

    function process(image) {

        const originalWidth =
            image.naturalWidth || image.width;

        const originalHeight =
            image.naturalHeight || image.height;

        if (!originalWidth || !originalHeight) {
            throw new Error("Image is not loaded.");
        }

        // STEP 1: Resize large photographs

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

        const sourceCanvas =
            document.createElement("canvas");

        sourceCanvas.width = width;
        sourceCanvas.height = height;

        const ctx =
            sourceCanvas.getContext("2d", {
                willReadFrequently: true
            });

        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);

        ctx.drawImage(
            image,
            0, 0,
            width, height
        );

        const rgba = ctx.getImageData(
            0, 0,
            width, height
        ).data;

        const count = width * height;

        // STEP 2: Convert to grayscale

        const gray = new Float32Array(count);

        for (let i = 0; i < count; i++) {

            const p = i * 4;

            gray[i] =
                0.299 * rgba[p] +
                0.587 * rgba[p + 1] +
                0.114 * rgba[p + 2];
        }

        // STEP 3: Calculate local brightness

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

        // STEP 4: Detect dark handwritten strokes

        const mask = new Uint8Array(count);

        const radius = Math.max(
            10,
            Math.round(
                Math.min(width, height) * 0.035
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

                const localMean = sum / area;

                const i = y * width + x;

                if (
                    localMean - gray[i] > 20 &&
                    gray[i] < 235
                ) {
                    mask[i] = 1;
                }
            }
        }

        // STEP 5: Find connected stroke regions

        const regionIds = new Int32Array(count);
        const queue = new Int32Array(count);
        const regions = [];

        let regionNumber = 0;

        for (let start = 0; start < count; start++) {

            if (
                mask[start] === 0 ||
                regionIds[start] !== 0
            ) {
                continue;
            }

            regionNumber++;

            let head = 0;
            let tail = 1;

            queue[0] = start;
            regionIds[start] = regionNumber;

            let minX = width;
            let minY = height;
            let maxX = 0;
            let maxY = 0;
            let touchesBorder = false;

            while (head < tail) {

                const current = queue[head++];

                const x = current % width;
                const y = Math.floor(current / width);

                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);

                if (
                    x < 3 || y < 3 ||
                    x >= width - 3 ||
                    y >= height - 3
                ) {
                    touchesBorder = true;
                }

                // Eight-neighbour connected components

                for (
                    let ny = Math.max(0, y - 1);
                    ny <= Math.min(height - 1, y + 1);
                    ny++
                ) {

                    for (
                        let nx = Math.max(0, x - 1);
                        nx <= Math.min(width - 1, x + 1);
                        nx++
                    ) {

                        const next = ny * width + nx;

                        if (
                            mask[next] === 1 &&
                            regionIds[next] === 0
                        ) {
                            regionIds[next] = regionNumber;
                            queue[tail++] = next;
                        }
                    }
                }
            }

            regions.push({
                id: regionNumber,
                size: tail,
                minX,
                minY,
                maxX,
                maxY,
                touchesBorder,
                centerX: (minX + maxX) / 2,
                centerY: (minY + maxY) / 2
            });
        }

        const minimumSize = Math.max(
            8,
            Math.floor(count * 0.000015)
        );

        const candidates = regions.filter(
            r => r.size >= minimumSize
        );

        if (candidates.length === 0) {
            throw new Error(
                "No clear handwritten character detected."
            );
        }

        // STEP 6: Choose the main character region

        const score = (r) => {

            const dx =
                (r.centerX - width / 2) / width;

            const dy =
                (r.centerY - height / 2) / height;

            const borderPenalty =
                r.touchesBorder ? 0.05 : 1;

            return (
                r.size * borderPenalty /
                (1 + 4 * (dx * dx + dy * dy))
            );
        };

        candidates.sort(
            (a, b) => score(b) - score(a)
        );

        const main = candidates[0];

        // Include nearby dots or marks

        const mainSize = Math.max(
            main.maxX - main.minX + 1,
            main.maxY - main.minY + 1
        );

        const margin = Math.max(
            12,
            mainSize * 0.4
        );

        const selected = candidates.filter(r => {

            if (r.id === main.id) return true;

            if (r.touchesBorder) return false;

            return (
                r.centerX >= main.minX - margin &&
                r.centerX <= main.maxX + margin &&
                r.centerY >= main.minY - margin &&
                r.centerY <= main.maxY + margin
            );
        });

        const selectedIds = new Set(
            selected.map(r => r.id)
        );

        let minX = width;
        let minY = height;
        let maxX = 0;
        let maxY = 0;

        for (const r of selected) {

            minX = Math.min(minX, r.minX);
            minY = Math.min(minY, r.minY);
            maxX = Math.max(maxX, r.maxX);
            maxY = Math.max(maxY, r.maxY);
        }

        // STEP 7: Create cleaned character mask

        const cleanCanvas =
            document.createElement("canvas");

        cleanCanvas.width = width;
        cleanCanvas.height = height;

        const cleanCtx =
            cleanCanvas.getContext("2d");

        const cleanData =
            cleanCtx.createImageData(width, height);

        for (let i = 0; i < count; i++) {

            const isStroke =
                selectedIds.has(regionIds[i]);

            const value = isStroke ? 255 : 0;
            const p = i * 4;

            cleanData.data[p] = value;
            cleanData.data[p + 1] = value;
            cleanData.data[p + 2] = value;
            cleanData.data[p + 3] = 255;
        }

        cleanCtx.putImageData(
            cleanData,
            0,
            0
        );

        // STEP 8: Crop, resize and center to 64x64

        const cropWidth = maxX - minX + 1;
        const cropHeight = maxY - minY + 1;

        const outputCanvas =
            document.createElement("canvas");

        outputCanvas.width = 64;
        outputCanvas.height = 64;

        const outputCtx =
            outputCanvas.getContext("2d");

        outputCtx.fillStyle = "black";

        outputCtx.fillRect(
            0,
            0,
            64,
            64
        );

        const resizeScale = Math.min(
            52 / cropWidth,
            52 / cropHeight
        );

        const newWidth = cropWidth * resizeScale;
        const newHeight = cropHeight * resizeScale;

        outputCtx.imageSmoothingEnabled = true;

        outputCtx.drawImage(
            cleanCanvas,
            minX,
            minY,
            cropWidth,
            cropHeight,
            (64 - newWidth) / 2,
            (64 - newHeight) / 2,
            newWidth,
            newHeight
        );

        // STEP 9: Convert to CNN pixel values

        const outputData = outputCtx.getImageData(
            0,
            0,
            64,
            64
        ).data;

        const pixels = new Float32Array(4096);

        for (let i = 0; i < 4096; i++) {

            pixels[i] =
                outputData[i * 4] / 255;
        }

        return {
            pixels: pixels,
            preview: outputCanvas.toDataURL("image/png")
        };
    }

    return {
        process
    };

})();
