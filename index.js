const express = require('express');
const { initModel, QAClient } = require("question-answering");

const app = express();
const port = process.env.PORT || 3000;

app.post('/questions', async (req, res) => {
  const text = req.body.text
  const question = req.body.question
  const model = await initModel({
    name: "distilbert-base-cased-distilled-squad",
    path: "https://qa-serving-f6hsrmjybq-uc.a.run.app/v1/models/cased",
    runtime: "remote"
  });
  const qaClient = await QAClient.fromOptions({ model });
  const answer = await qaClient.predict(question, text);
  console.log(answer);

  res.json({
    text: text,
    question: question,
    answer: answer.text,
    score: answer.score
  });
});

app.listen(port, () => {
  console.log(`Server running on port: ${port}`);
  console.log('Press CTRL + C to quit');
})