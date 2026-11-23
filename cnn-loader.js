
/*
 Bengali Handwriting Recognition
 CNN Model Loader

 Model: 13-layer CNN
 Input: 64 x 64 grayscale
 Output: 60 Bengali characters
*/

window.BanglaCNN = (() => {

    let model = null;
    let labels = null;
    let loadingPromise = null;

    async function fetchFile(path, type) {

        const response = await fetch(path);

        if (!response.ok) {
            throw new Error(
                "Failed to load " + path +
                " (HTTP " + response.status + ")"
            );
        }

        if (type === "json") {
            return response.json();
        }

        return response.arrayBuffer();
    }

    async function initialize() {

        if (!window.tf) {
            throw new Error(
                "TensorFlow.js library is missing."
            );
        }

        await tf.ready();

        const manifest = await fetchFile(
            "./browser_model/layers.json",
            "json"
        );

        const binary = await fetchFile(
            "./browser_model/weights.bin",
            "binary"
        );

        const loadedLabels = await fetchFile(
            "./bangla_labels_60.json",
            "json"
        );

        if (!Array.isArray(loadedLabels) ||
            loadedLabels.length !== 60) {
            throw new Error(
                "Expected 60 Bengali character labels."
            );
        }

        const cnn = tf.sequential();

        try {

            cnn.add(tf.layers.inputLayer({
                inputShape: manifest.inputShape
            }));

            for (const info of manifest.layers) {

                const cfg = info.config;
                let layer;

                switch (info.type) {

                    case "Conv2D":
                        layer = tf.layers.conv2d({
                            filters: cfg.filters,
                            kernelSize: cfg.kernelSize,
                            strides: cfg.strides,
                            padding: cfg.padding,
                            activation: cfg.activation,
                            useBias: cfg.useBias,
                            dilationRate: cfg.dilationRate
                        });
                        break;

                    case "BatchNormalization":
                        layer = tf.layers.batchNormalization({
                            axis: cfg.axis,
                            epsilon: cfg.epsilon,
                            momentum: cfg.momentum,
                            center: cfg.center,
                            scale: cfg.scale
                        });
                        break;

                    case "MaxPooling2D":
                        layer = tf.layers.maxPooling2d({
                            poolSize: cfg.poolSize,
                            strides: cfg.strides,
                            padding: cfg.padding
                        });
                        break;

                    case "Flatten":
                        layer = tf.layers.flatten();
                        break;

                    case "Dense":
                        layer = tf.layers.dense({
                            units: cfg.units,
                            activation: cfg.activation,
                            useBias: cfg.useBias
                        });
                        break;

                    case "Dropout":
                        layer = tf.layers.dropout({
                            rate: cfg.rate
                        });
                        break;

                    default:
                        throw new Error(
                            "Unsupported layer: " + info.type
                        );
                }

                cnn.add(layer);

                const tensors = [];

                try {

                    for (const spec of info.weights) {

                        if (spec.byteOffset % 4 !== 0 ||
                            spec.byteOffset +
                            spec.length * 4 >
                            binary.byteLength) {
                            throw new Error(
                                "Invalid weight data: " +
                                info.name
                            );
                        }

                        const values = new Float32Array(
                            binary,
                            spec.byteOffset,
                            spec.length
                        );

                        tensors.push(
                            tf.tensor(
                                values,
                                spec.shape,
                                "float32"
                            )
                        );
                    }

                    if (tensors.length > 0) {
                        layer.setWeights(tensors);
                    }

                } finally {
                    tensors.forEach(t => t.dispose());
                }
            }

            if (cnn.outputShape[1] !== 60) {
                throw new Error(
                    "Model must have 60 output classes."
                );
            }

            model = cnn;
            labels = loadedLabels;

            console.log(
                "Bengali CNN loaded successfully!"
            );

            return true;

        } catch (error) {

            cnn.dispose();
            throw error;
        }
    }

    async function load() {

        if (model) return true;

        if (!loadingPromise) {

            loadingPromise = initialize().catch(
                error => {
                    loadingPromise = null;
                    throw error;
                }
            );
        }

        return loadingPromise;
    }

    async function predict(pixels) {

        await load();

        if (!pixels || pixels.length !== 4096) {
            throw new Error(
                "Expected a 64x64 grayscale image."
            );
        }

        const values = Float32Array.from(pixels);

        if (!values.every(Number.isFinite)) {
            throw new Error("Invalid pixel values.");
        }

        const input = tf.tensor4d(
            values,
            [1, 64, 64, 1],
            "float32"
        );

        let output;

        try {

            output = model.predict(input);

            const scores = Array.from(
                await output.data()
            );

            const results = scores.map(
                (score, index) => ({
                    classId: index + 1,
                    character: labels[index],
                    score: score
                })
            );

            results.sort(
                (a, b) => b.score - a.score
            );

            return {
                character: results[0].character,
                classId: results[0].classId,
                confidence: results[0].score,
                top3: results.slice(0, 3)
            };

        } finally {

            input.dispose();

            if (output) {
                output.dispose();
            }
        }
    }

    return {
        load,
        predict
    };

})();
