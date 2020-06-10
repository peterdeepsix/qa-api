// Express Deps
const express = require('express');
const cors = require('cors')
const swaggerUi = require('swagger-ui-express');
const spec = require('../swagger');

// ML Deps
const { initModel, QAClient } = require("question-answering");
const use = require('@tensorflow-models/universal-sentence-encoder');

// Init Express
const app = express();

// Listen on PORT env var
const port = process.env.PORT || 3000;

// Use built in body parsing middleware
app.use(express.json());

// Init CORS options
app.options('*', cors())

// Configure CORS for all routes
app.use(cors());

// Bypass CORS for all origins
// Remove once service is internal
app.use(function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  next();
});

// Express Route - Wildcard All
app.get("/", function(req, res) {
  res.send("Question Answer API");
})

// Express Route - Swagger Documentation
app.use('/docs', swaggerUi.serve, swaggerUi.setup(spec));
/**
* @swagger
* /questions:
*   post:
*     tags:
*       — Questions
*     summary: This should return a list of similar answers.
*     description: This is where you can give some background as to why this route is being created or perhaps reference a ticket number.
*     consumes:
*       — application/json
*     parameters:
*       — name: body
*       in: body
*       schema:
*         type: object
*         properties:
*           flavor:
*           type: string
*     responses: 
*       200:
*         description: Receive back flavor and flavor Id.
*/
app.post('/questions', async (req, res) => {

  // Init req body
  const input = req.body

  // Init BERT model
  const model = await initModel({
    name: "distilbert-base-cased-distilled-squad",
    path: "https://qa-serving-f6hsrmjybq-uc.a.run.app/v1/models/cased",
    runtime: "remote"
  });

  // Init BERT QA client
  const qaClient = await QAClient.fromOptions({ model });

  // Return answer from QA client based on req.body
  const answer = await qaClient.predict(input.question, input.text);

  // Universal Sentence Encoder - use
  const input_threshold = input.confidence_threshold

  // Init sentences from req.body
  const sentences = [
    answer.text,
    input.test_answer,
    input.known_answer
  ];

  // Dot function for a sentance
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

  // Compare the similarity of sentences
  const  similarity = (a, b) => {
    var magnitudeA = Math.sqrt(dot(a, a));
    var magnitudeB = Math.sqrt(dot(b, b));
    if (magnitudeA && magnitudeB)
      return dot(a, b) / (magnitudeA * magnitudeB);
    else return false
  }

  // Calculate the cosine similarity matrix from the sentence embeddings
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

  // Form groups of sentences based on the similarity matrix
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

    return return_groups;
  }

  // Group embedded sentences into similar groups
  const get_similarity = async (embeddings) => {

    let matrix = await cosine_similarity_matrix(embeddings.arraySync());

    let formed_groups = await form_groups(matrix);

    let render_groups = [];

    for(let i in formed_groups){
      let sub_group = []
      for(let j in formed_groups[i]){
        sub_group[j] = sentences[ formed_groups[i][j] ]
      }
      render_groups[i] = sub_group
    }
    
    return render_groups
  }

  // Generate the final api response for /questions
  const final_response = await use.load().then(async model => {
    response = await model.embed(sentences).then(async embeddings => {
      embeddings.print(true /* verbose */);

      let response_groups = await get_similarity(embeddings)

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

app.post('/similarity', async (req, res) => {

  // Init req body
  const input = req.body

  // Universal Sentence Encoder - use
  const input_threshold = input.input_threshold

  // Init sentences from req.body
  const sentences = input.sentences;

  // Dot function for a sentance
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

  // Compare the similarity of sentences
  const  similarity = (a, b) => {
    var magnitudeA = Math.sqrt(dot(a, a));
    var magnitudeB = Math.sqrt(dot(b, b));
    if (magnitudeA && magnitudeB)
      return dot(a, b) / (magnitudeA * magnitudeB);
    else return false
  }

  // Calculate the cosine similarity matrix from the sentence embeddings
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

  // Form groups of sentences based on the similarity matrix
  const form_groups = async (cosine_similarity_matrix) => {
    let dict_keys_in_group = {};
    let groups = [];
    let groupsSimScore = []

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
              groupsSimScore.push([]);
            }
            groups[group_num].push(i);
            groups[group_num].push(j);
            groupsSimScore[group_num].push({
              i: i,
              j: j,
              sim_score: sim_score,
            });

            console.log("i is similar to j", i, j)
            console.log("sim_score", sim_score)
            console.log("groups", groups)
            console.log("groupsSimScore", groupsSimScore)
            
          }
        }
      }
    }

    let return_groups = [];

    for(var i in groups){
      return_groups.push(Array.from(new Set(groups[i])));
    }

    let return_object = {
      groups: return_groups,
      groups_edge_score: groupsSimScore,
    }

    return return_object;
  }

  // Group embedded sentences into similar groups
  const get_similarity = async (embeddings) => {

    let matrix = await cosine_similarity_matrix(embeddings.arraySync());

    let return_object = await form_groups(matrix);

    let formed_groups = return_object.groups

    let formed_groups_sim_score = return_object.groups_edge_score

    let render_groups = [];

    let render_groups_sim_score = [];
    

    for(let i in formed_groups){
      let sub_group = []
      for(let j in formed_groups[i]){
        sub_group[j] = sentences[ formed_groups[i][j] ]
      }
      render_groups[i] = sub_group
      render_groups_sim_score[i] = {
        group: sub_group,
        // groups_edge_score: formed_groups_sim_score,
      }
    }
    
    for(let i in formed_groups){
      let sub_group = []
      for(let j in formed_groups[i]){
        sub_group[j] = sentences[ formed_groups[i][j] ]
      }
      render_groups[i] = sub_group
      render_groups_sim_score[i] = {
        ...render_groups_sim_score[i],
        groups_edge_score: formed_groups_sim_score[i],
      }
    }

    return render_groups_sim_score
  }

  // Generate the final api response for /questions
  const similar_groups = await use.load().then(async model => {
    response = await model.embed(sentences).then(async embeddings => {
      embeddings.print(true /* verbose */);

      let response_object = await get_similarity(embeddings)

      return response_object
    });
    
  return response
  });

  res.json({
    similar_groups: similar_groups
  });
});

// Embed Questions
app.post('/embed', async (req, res) => {

  // Init req body
  const input = req.body

  // Init sentences from req.body
  const sentences = input.sentences

  const saved_embeddings = await use.load().then(async model => {
    // use the model here
    return await model.embed(sentences).then(async embeddings => {
      // `embeddings` is a 2D tensor consisting of the 512-dimensional embeddings for each sentence.
      // So in this example `embeddings` has the shape [2, 512].
      let shape = await embeddings.shape
      let data = await embeddings.data()
      let saved = {data: data, shape: shape}
      return saved
    });
  })

  res.json({
    saved_embeddings: saved_embeddings
  });
});



//Encode Sentences
app.post('/encode', async (req, res) => {

  // Init req body
  const input = req.body

  // Init sentences from req.body
  const sentences = input.sentences

  // Encode Sentences
  let encoded_sentences =  await use.loadTokenizer().then( async tokenizer => {
    return  tokenizer.encode(sentences[0]);
  });

  res.json({
    encoded_sentences: encoded_sentences
  });
});


// Express listen on port
app.listen(port, () => {
  console.log(`Server running on port: ${port}`);
  console.log('Press CTRL + C to quit');
})