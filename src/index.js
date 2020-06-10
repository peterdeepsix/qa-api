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
    let groups_edges =[]

    // Based on the input_threshold add similar groups into groups, and add all the similarity comparisons to groups_edges
    for(let i=0; i<cosine_similarity_matrix.length; i++){
      var this_row = cosine_similarity_matrix[i];
      for(let j=i; j<this_row.length; j++){
        
        // Make sure we are comparing different sentences and asign the similarity score.
        if(i!=j){
          let sim_score = cosine_similarity_matrix[i][j];
          
          // Check to see if the similarity score is greater than the input_threshold.
          if(sim_score > input_threshold){
            let group_num;
            let group_object ={
              i: i,
              j: j,
              sim_score: sim_score
            };

            // Check to see if the sentence comparison exists in the dictionary of group keys.
            // Then set the group_num to the length of the current groups and assign it to the group_num.
            // Otherwise assign the group_num to the current dictionary entry of group keys.
            if(!(i in dict_keys_in_group)){
              group_num = groups.length;
              dict_keys_in_group[i] = group_num;
            }else{
              group_num = dict_keys_in_group[i];
            }

            // Check to see if the the second level sentence comparison exists and assign it to the dictionary of group keys.
            if(!(j in dict_keys_in_group)){
              dict_keys_in_group[j] = group_num;
            }

            // Check to see if the groups is empty, and inititalizes.
            if(groups.length <= group_num){
              groups.push([]);
              groups_edges.push([]);
            }
            
            // Addeds the sentence to the groups.
            groups[group_num].push(i);

            // Adds the second level sentence to the groups.
            groups[group_num].push(j);

            // Adds the the compariosn edge to the groups_edges.
            groups_edges[group_num].push(group_object);

            console.log("group_object", group_object)
            console.log("groups", groups)
            console.log("groups_edges", groups_edges)
            
          }
        }
      }
    }

    let groups_set = [];

    // Use "new Set to make sure a sentence exists only once in the groups array"
    for(var i in groups){
      groups_set.push(Array.from(new Set(groups[i])));
    }
    
    // Return both the set of unique groups and the array of groups similarity edges.
    return [groups_set, groups_edges];
  }

  // Group embedded sentences into similar groups
  const get_similarity = async (embeddings) => {

    // Process the senteces similarity.
    let matrix = await cosine_similarity_matrix(embeddings.arraySync());

    // Massage the sentences similarity matrix into unique groups and groups_edges.
    const [groups_set, groups_edges] = await form_groups(matrix);

    return [groups_set, groups_edges]
  }

  // Generate the response for /similarity
  const similar_groups = await use.load().then(async model => {
    response = await model.embed(sentences).then(async embeddings => {
      embeddings.print(true /* verbose */);

      const [groups_set, groups_edges]= await get_similarity(embeddings)

      let sentences_set = [];
    
      for(let i in groups_set){
        let sub_group = []
        for(let j in groups_set[i]){
          sub_group[j] = sentences[ groups_set[i][j] ]
        }
        sentences_set[i] = sub_group
      }

      return [sentences_set, groups_edges]
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