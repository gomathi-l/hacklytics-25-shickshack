// Step 1: Import the tools we need
const express = require('express');
const mongoose = require('mongoose');
const dotenv = require('dotenv');

// Step 2: Load environment variables (secrets)
dotenv.config();

// Step 3: Create an Express app
const app = express();

// Step 4: Connect to MongoDB
mongoose.connect(process.env.MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('Connected to MongoDB!'))
  .catch((err) => console.log('Error connecting to MongoDB:', err));

// Step 5: Start the server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});


const todoRoutes = require('./routes/todoRoutes');

// Middleware to parse JSON
app.use(express.json());

// Use the todo routes
app.use('/todos', todoRoutes);