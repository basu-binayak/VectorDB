# Vector Databases
A **vector database** is a type of database that specializes in storing, searching, and retrieving data represented as **vectors**. Vectors are mathematical objects that are essentially lists of numbers, used to represent information in a way that computers can process efficiently. To break it down, let's explore the concept in a simple, understandable way.

### What is a vector?

Imagine you have a **profile** of a person, which includes their age, height, weight, and income. You can represent this information as a list of numbers (for example, `[28, 170, 75, 50000]`), where each number corresponds to one of these features. This list is a vector. In the same way, any kind of data—whether it’s text, images, or audio—can be transformed into vectors using mathematical techniques.

### Why are vectors useful?

Vectors allow computers to process and compare different types of data in a uniform way. For example, when you look for "similar" items (such as similar movies, products, or even similar faces), vectors make this possible by representing complex objects (like a movie's genre, style, and reviews) as numbers. Vectors enable algorithms to calculate how close or similar two objects are.

### How is a vector database different from a traditional database?

In a **traditional database**, data is often stored in a structured way, like in tables with rows and columns (similar to a spreadsheet). You can search this data using specific values. For example, you might ask a traditional database to find all people who are 28 years old or earn $50,000 a year.

A **vector database**, on the other hand, allows you to search for things based on **similarity** rather than specific values. Instead of asking "who is 28 years old?", you might ask "who is most similar to this person based on all of their attributes?" The database will look for other vectors (lists of numbers) that are close to the vector you’re searching for. This is especially useful in applications like:

- **Search engines**: Finding similar documents or web pages.
- **Recommendation systems**: Suggesting similar movies, songs, or products.
- **Image recognition**: Finding pictures that look alike.
- **Natural language processing (NLP)**: Comparing texts to find related meanings or similar sentences.

### How does a vector database work?

1. **Data Encoding (Vectorization)**: First, the raw data (like text, images, or audio) is converted into vectors. For example, if you want to store pictures of cats and dogs, you use special machine learning models to convert each picture into a list of numbers that represent its features (like shape, color, or texture).

2. **Storing Vectors**: These vectors are stored in the vector database. The database is optimized to handle millions or billions of vectors.

3. **Searching by Similarity**: When you want to find similar items, the vector database uses mathematical techniques to measure the distance between vectors (using something called **cosine similarity** or **Euclidean distance**, for example). The closer two vectors are, the more similar the items they represent are considered to be.

4. **Efficient Retrieval**: Vector databases are designed to quickly retrieve the most similar vectors, even if there are millions of them. Traditional databases would struggle to do this efficiently because they aren’t built for similarity-based searches.

### Real-world Examples of Vector Databases

1. **Search Engines**: Google or Bing can search not just for exact keywords but also for results related in meaning, thanks to vectors. If you search for “best laptops for students,” the search engine will also show results for similar queries like “affordable laptops for college” because these queries are represented by similar vectors.

2. **Recommendation Systems**: Netflix or Spotify recommends shows, movies, or songs that are similar to what you've watched or listened to before. They use vector databases to represent the characteristics of the content (genre, actors, or style) as vectors and then find similar ones to recommend.

3. **Chatbots**: When you interact with a chatbot that understands human language (like customer service bots), it uses vectors to understand the meaning of your sentences. For instance, "I need help with my account" and "Can you assist with my profile?" might be treated as similar because the chatbot's vector database can measure how close these sentences are in meaning.

### Why is the use of vector databases growing?

As we move deeper into the world of **artificial intelligence (AI)**, we need databases that can handle **unstructured data** like text, images, and videos. Traditional databases work well with structured data (like numbers and categories), but when it comes to finding similarities or patterns in more complex data types, vector databases shine.

For example:
- AI applications like image recognition or language models require comparing complex data to find similarities (like finding similar faces in a photo or identifying the topic of a document).
- Vectors allow you to search through massive amounts of data quickly and efficiently, which is essential as we deal with more data than ever before.

### In summary:
A **vector database** is specialized for handling and searching large amounts of data that has been transformed into vectors. It's especially useful in AI-driven tasks, such as finding similarities between objects (like images, text, or even people) and making recommendations. With the growing use of AI and machine learning, vector databases are becoming more popular because they are designed to manage and search unstructured data in a way that traditional databases cannot.

# How a Vector Database works?
To provide a **step-by-step analysis** of how a **vector store** works—from receiving a document to storing it as a vector—we’ll break down the entire process. This includes data preparation, transformation, and storage in a vector database (or vector store). Here's how the journey typically looks:

### 1. **Receiving the Document**
The process begins when a **document** (or any other form of data like text, image, or audio) is submitted to the system. A document could be an article, a product description, a user query, or any form of unstructured data.

#### Example: 
Imagine we are working with a text document, like a product description for an online store.

- **Input**: "This is a high-quality leather jacket, perfect for winter."

### 2. **Preprocessing the Document**
Before converting the document into a vector, it may need to undergo **preprocessing**. This step ensures that the data is clean and ready to be vectorized. Preprocessing can involve different steps depending on the type of data (text, images, etc.).

For a **text document**, preprocessing might include:
- **Tokenization**: Breaking the document into individual words or sentences.
- **Lowercasing**: Converting all text to lowercase to avoid treating "Jacket" and "jacket" as different words.
- **Removing Stop Words**: Filtering out common words like "the," "is," or "for" that don’t carry much meaning.
- **Stemming/Lemmatization**: Reducing words to their base form (e.g., "running" becomes "run").

#### Example after preprocessing:
- **Processed text**: "high-quality leather jacket perfect winter"

### 3. **Vectorization (Converting Document to a Vector)**
Once the document is preprocessed, the next step is to convert it into a **vector**. This process is called **vectorization**, and it involves transforming the document into a list of numbers that mathematically represent the features of the data.

For text documents, vectorization is often done using:
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: A method that assigns a numerical value to each word in the document, based on how important that word is in the overall dataset.
- **Word Embeddings (e.g., Word2Vec, GloVe)**: These techniques represent words in a continuous vector space, capturing semantic relationships between words.
- **Transformer Models (e.g., BERT, GPT)**: More advanced techniques that can capture the context and meaning of words in relation to each other.

#### Example:
Let’s say we use a word embedding model like Word2Vec. It converts the processed text into a vector that might look like this:

- **Vector**: `[0.25, 0.83, 0.11, -0.56, 0.67, ...]` (This is a simplified representation; in reality, vectors can be hundreds of dimensions long.)

### 4. **Storing the Vector in the Vector Database**
Once the document is converted into a vector, this vector (along with some metadata about the document) is stored in the **vector database**.

The vector database specializes in handling high-dimensional vectors efficiently and supports similarity searches. When storing the vector, the database often retains:
- The **document's original content** or a reference to the original document.
- The **vector** that represents the document.
- **Metadata** like timestamps, categories, or other additional information.

#### Example:
- **Stored Entry**: 
    - Original Document: "This is a high-quality leather jacket, perfect for winter."
    - Vector: `[0.25, 0.83, 0.11, -0.56, 0.67, ...]`
    - Metadata: `{category: "clothing", timestamp: "2024-10-23"}`

### 5. **Indexing the Vector for Efficient Searching**
In addition to storing the vector, the vector database needs to **index** it. Indexing is what enables the database to perform fast similarity searches later on.

Popular indexing techniques for vector databases include:
- **Approximate Nearest Neighbor (ANN)** methods, such as **HNSW (Hierarchical Navigable Small World)** or **LSH (Locality-Sensitive Hashing)**. These methods make it possible to search for vectors that are "close" to a query vector very efficiently, even when dealing with millions or billions of vectors.

### 6. **Similarity Search and Querying**
Once the vector is stored and indexed, the vector store can now perform searches based on **similarity**.

For example, if you want to search for documents similar to another leather jacket description, you can:
- **Convert the query document** into a vector using the same vectorization process.
- **Search for similar vectors** in the vector store using a similarity metric, such as **cosine similarity** or **Euclidean distance**.

The database will return the most similar vectors and their associated documents.

#### Example Query:
- Query: "Find jackets similar to: 'Stylish winter coat, made of wool, very warm.'"
- This query will be vectorized, and the vector store will return vectors of documents that are closest in meaning to this description.

### 7. **Returning the Results**
Once the vector store identifies the most similar vectors, it will return the **corresponding documents** to the user. The results may include a list of relevant documents, sorted by how similar they are to the query.

#### Example Output:
- Similar results: 
    1. "High-quality leather jacket, perfect for winter" (cosine similarity score: 0.95)
    2. "Woolen winter coat, extremely cozy and warm" (cosine similarity score: 0.92)
    3. "Lightweight jacket, suitable for fall" (cosine similarity score: 0.85)

### Recap of the Step-by-Step Process:
1. **Document Reception**: Receive the document (e.g., a text description).
2. **Preprocessing**: Clean and prepare the document (tokenize, remove stop words, etc.).
3. **Vectorization**: Convert the document into a numerical vector using techniques like Word2Vec, BERT, or TF-IDF.
4. **Storing the Vector**: Store the vector along with metadata in the vector database.
5. **Indexing**: Create an index for efficient similarity searching.
6. **Similarity Search**: When querying, convert the query into a vector and find the closest vectors in the database.
7. **Return Results**: Retrieve and present the most similar documents.

This process makes it possible for vector stores to power **AI-driven applications** like recommendation systems, semantic search engines, and more by enabling the efficient storage and retrieval of data based on **similarity** rather than exact matches.
<img src="./Images/vector.png">

# What is the difference between EMBEDDINGS and VECTORS?
### What Are Embeddings?

**Embeddings** are a type of representation for data—usually in the form of **dense vectors**—that map complex, unstructured data like words, sentences, images, or user profiles into a **numerical space**. These embeddings capture **relationships and similarities** in the data so that similar items are represented by vectors that are close to each other in that space.

In simpler terms:
- **Embeddings** convert real-world data into a list of numbers, where the **meaning** or **context** of the data is embedded into these numbers.
- They are commonly used in **machine learning** to represent complex data in a way that makes it easier for algorithms to understand, compare, and find patterns.

#### Example:
If you have two words, "king" and "queen," an embedding model might represent them as:
- "king" → `[0.7, 0.5, 0.3]`
- "queen" → `[0.7, 0.4, 0.4]`

These vectors are similar because the words "king" and "queen" have related meanings.

### What Are Vectors?

**Vectors** are simply lists of numbers (sometimes called arrays) that represent points in a multi-dimensional space. They are a general mathematical concept, widely used in many fields beyond machine learning and data science, such as physics, economics, and graphics.

In the context of embeddings:
- Vectors are the **numerical representation** produced by embedding techniques.
- They serve as a format for representing data in machine learning models or vector databases.

Vectors can have any dimension (for example, a 3-dimensional vector `[0.7, 0.5, 0.3]` or a 300-dimensional vector depending on the model).

### Key Differences Between Embeddings and Vectors

1. **Purpose**:
   - **Embeddings** are designed specifically to capture relationships or patterns in data, like **semantic meaning** for words, **similarity** between images, or other forms of latent information.
   - **Vectors** are the format or structure used to represent data mathematically. They can be generated from any kind of process (not necessarily embeddings) and don't inherently capture relationships between data unless they are generated from an embedding process.

2. **Creation Process**:
   - **Embeddings** are generated using **machine learning models** that have been trained on large datasets to capture relationships. Popular techniques include:
     - **Word embeddings** like **Word2Vec**, **GloVe**, or **FastText** for text.
     - **Image embeddings** using **convolutional neural networks (CNNs)**.
     - **Sentence embeddings** using models like **BERT** or **GPT**.
   - **Vectors**, on the other hand, can be created from simple mathematical processes like assigning random numbers or basic feature extraction. They are more generic and do not necessarily carry any contextual or meaningful information unless they come from an embedding process.

3. **Context and Meaning**:
   - **Embeddings** are specifically built to preserve the **semantic meaning** or **similarity** between data points. For example, embeddings will place words with similar meanings (like "car" and "automobile") close to each other in the vector space.
   - **Vectors**, in a general sense, don’t inherently represent similarity or meaning unless they’ve been produced by some meaningful process (like embeddings). For example, a vector representing physical forces or geographical coordinates does not inherently capture "similarity" between data points in a human-understandable way.

### Example: Text Embeddings vs. Arbitrary Vectors
Imagine you have two sentences:
- **Sentence 1**: "The cat sat on the mat."
- **Sentence 2**: "A dog lay on the rug."

If you use a language model to generate embeddings for these sentences, their vectors might look like this:
- Sentence 1 → `[0.8, 0.2, 0.9, -0.1, 0.4, ...]`
- Sentence 2 → `[0.7, 0.3, 0.8, -0.2, 0.5, ...]`

These vectors will be **close to each other** in the embedding space because the sentences have similar meanings (both describe animals resting on surfaces).

Now, imagine you just randomly assign vectors to these sentences:
- Sentence 1 → `[2.3, 1.5, -0.7, 0.4]`
- Sentence 2 → `[-1.2, 3.4, 0.9, -2.7]`

These **arbitrary vectors** don't carry any meaning, and their values were assigned without regard to the content of the sentences. In this case, the vectors don't represent anything semantically useful.

### Use Cases for Embeddings

1. **Search Engines**: Search queries can be embedded into vectors, and then the vector representing the query can be compared to vectors representing web pages or documents. This allows the search engine to find documents that are **semantically similar** to the query, even if they don’t contain the exact keywords.

2. **Recommendation Systems**: Embeddings are used to recommend content by representing user preferences, behaviors, or content attributes as vectors. Similar vectors imply that the user would be interested in similar items.

3. **Natural Language Processing (NLP)**: Embeddings are essential in many NLP tasks like machine translation, text classification, or sentiment analysis, as they capture the **meaning** of words or sentences.

4. **Image and Video Recognition**: In computer vision, embeddings are used to represent images or video frames in a way that similar-looking objects are mapped to vectors that are close together in the embedding space.

### Summary:
- **Embeddings** are specialized **dense vectors** that represent complex data like text, images, or user preferences in a way that captures **relationships** and **similarities**.
- **Vectors** are general lists of numbers and can represent anything. In the case of embeddings, vectors are the output that carry the embedded meaning or similarity of the original data.

Embeddings are essentially a **type of vector** created using machine learning to capture meaningful patterns in the data, while not all vectors are embeddings.