const express = require('express');
const { QAClient } = require("question-answering");

const app = express();
const port = process.env.PORT || 3000;

const text = `
  Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season.
  The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
  As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.
`;

const question = "Who won the Super Bowl?";

const predict = async () => {
    const qaClient = await QAClient.fromOptions();
    const answer = await qaClient.predict(question, text);
    console.log(answer);
    return answer
}

app.get('/', (req, res) => {
  const answer = predict()
  res.json({
    answer: answer.text,
    score: answer.score
  });
});

app.listen(port, () => {
  console.log(`Server running on port: ${port}`);
  console.log('Press CTRL + C to quit');
})