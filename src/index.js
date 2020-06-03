const express = require('express');
const cors = require('cors')

const { initModel, QAClient } = require("question-answering");
const use = require('@tensorflow-models/universal-sentence-encoder');

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
  console.log(input.test_answer);
  console.log(input.known_answer);
  console.log(answer.text);
  console.log(answer.score);
  console.log(input.confidence_threshold);



  // Universal Sentence Encoder - use
const input_threshold = input.confidence_threshold

const sentences = [
  answer.text,
  input.test_answer,
  input.known_answer
];

const dot = (a, b) => {
    var hasOwnProperty = Object.prototype.hasOwnProperty;
    var sum = 0;
    for (var key in a) {
      if (hasOwnProperty.call(a, key) && hasOwnProperty.call(b, key)) {
        sum += a[key] * b[key]
      }
    }
    return sum
  }

const  similarity = (a, b) => {
  var magnitudeA = Math.sqrt(dot(a, a));
  var magnitudeB = Math.sqrt(dot(b, b));
  if (magnitudeA && magnitudeB)
    return dot(a, b) / (magnitudeA * magnitudeB);
  else return false
}

const cosine_similarity_matrix = (matrix) => {
  let cosine_similarity_matrix = [];
  for(let i=0;i<matrix.length;i++){
    let row = [];
    for(let j=0;j<i;j++){
      row.push(cosine_similarity_matrix[j][i]);
    }
    row.push(1);
    for(let j=(i+1);j<matrix.length;j++){
      row.push(similarity(matrix[i],matrix[j]));
    }
    cosine_similarity_matrix.push(row);
  }
  return cosine_similarity_matrix;
}

const form_groups = async (cosine_similarity_matrix) => {
  let dict_keys_in_group = {};
  let groups = [];

  for(let i=0; i<cosine_similarity_matrix.length; i++){
    var this_row = cosine_similarity_matrix[i];
    for(let j=i; j<this_row.length; j++){
      if(i!=j){
        let sim_score = cosine_similarity_matrix[i][j];

        if(sim_score > input_threshold){

          let group_num;

          if(!(i in dict_keys_in_group)){
            group_num = groups.length;
            dict_keys_in_group[i] = group_num;
          }else{
            group_num = dict_keys_in_group[i];
          }
          if(!(j in dict_keys_in_group)){
            dict_keys_in_group[j] = group_num;
          }

          if(groups.length <= group_num){
            groups.push([]);
          }
          groups[group_num].push(i);
          groups[group_num].push(j);
        }
      }
    }
  }
  let return_groups = [];
  for(var i in groups){
    return_groups.push(Array.from(new Set(groups[i])));
  }

  console.log("return_groups", return_groups);
  return return_groups;
}

const get_similarity = async (embeddings) => {
  console.log("embeddings", embeddings);

  let matrix = await cosine_similarity_matrix(embeddings.arraySync());
  console.log("matrix", matrix);

  let formed_groups = await form_groups(matrix);
  console.log("formed_groups", formed_groups);

  let render_groups = [];

  for(let i in formed_groups){
    let sub_group = []
    for(let j in formed_groups[i]){
      sub_group[j] = sentences[ formed_groups[i][j] ]
      console.log(sentences[ formed_groups[i][j] ])
    }
    console.log('sup_group', sub_group)
    render_groups[i] = sub_group
  }
  for(let i in render_groups){
    console.log("render_group", render_groups[i]);
  }
  
  return render_groups
}

const final_response = await use.load().then(async model => {
  response = await model.embed(sentences).then(async embeddings => {
    embeddings.print(true /* verbose */);
    let response_groups = await get_similarity(embeddings)
    console.log("response_groups", response_groups);
    return response_groups
  });
 return response
});

  res.json({
    text: input.text,
    question: input.question,
    answer: answer.text,
    score: answer.score,
    final_response:final_response
  });
});

app.listen(port, () => {
  console.log(`Server running on port: ${port}`);
  console.log('Press CTRL + C to quit');
})