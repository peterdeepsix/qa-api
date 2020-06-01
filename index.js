const express = require('express');
const cors = require('cors')

const { initModel, QAClient } = require("question-answering");

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

app.options('*', cors())
app.use(cors());

app.use(function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  next();
});


app.post('/questions', async (req, res) => {
  const input = req.body
  const model = await initModel({
    name: "distilbert-base-cased-distilled-squad",
    path: "https://qa-serving-f6hsrmjybq-uc.a.run.app/v1/models/cased",
    runtime: "remote"
  });

  const qaClient = await QAClient.fromOptions({ model });
  const answer = await qaClient.predict(input.question, input.text);

  console.log(input.text);
  console.log(input.question);
  console.log(answer.text);
  console.log(answer.score);

  res.json({
    text: input.text,
    question: input.question,
    answer: answer.text,
    score: answer.score
  });
});

app.listen(port, () => {
  console.log(`Server running on port: ${port}`);
  console.log('Press CTRL + C to quit');
})