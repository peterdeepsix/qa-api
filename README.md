# qa-api

# Transformers Library
https://github.com/huggingface/transformers
https://huggingface.co/transformers/

# Pretrained BERT Model
https://medium.com/huggingface/distilbert-8cf3380435b5

https://huggingface.co/distilbert-base-cased-distilled-squad

# Universal Sentence Encoder
https://medium.com/acing-ai/what-is-cosine-similarity-matrix-f0819e674ad1#:~:text=Cosine%20similarity%20is%20a%20metric,in%20a%20multi%2Ddimensional%20space.&text=Mathematically%2C%20if%20'a'%20and,the%20angle%20between%20the%20two.

https://tfhub.dev/google/universal-sentence-encoder/4


# TensorFlow Serving
https://github.com/tensorflow/serving
https://www.tensorflow.org/tfx/guide/serving

# Process

Models "distilbert-base-cased-distilled-squad" and "google/universal-sentence-encoder" live in a cloud storage bucket.

Tensorflow serving docker container consumes the latest model and deploys it as a versioned servable.

API container consumes the servables from TFServing to predict.

QA function returns the AI Generated Answer based on the BERT model and the input text and question.

Use universal sentence encoder to embed the array of sentences.

Get similarity between all of the sentences.

Use Cosine similarity to generate a matrix of probabilities. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space

With the matrix of probabilities sort the sentences into groups when they are more similar than the input threshold.

Return and display the groups of sorted sentences.


