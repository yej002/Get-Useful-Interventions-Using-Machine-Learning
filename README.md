# team-neu-yej002-zhijingt-ziywang50

The goal of the project is to build a prototype of an AI-based tool/machine learning tool that public sector organizations of the Beam Group company can use to predict service or intervention needs based on client characteristics and to predict the probabilities of Return to Work if the interventions are taken by the client.  

# Demo
[![demo](https://img.youtube.com/vi/YgJ39xx93ls/0.jpg)](https://www.youtube.com/watch?v=YgJ39xx93ls)



## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

- [Python](https://www.python.org/downloads/) (for the backend)
- [Node.js and npm](https://nodejs.org/en/download/) (for the frontend)

Note: The development was taken in the conda environment, if you don't have the necessary modules like sklearn, pandas, numpy, etc., please install them by typing the commands below in your terminal:
```bash
   pip install sklearn
   pip install pandas
   pip install numpy
```

or you can activate your conda environment by: 
```bash
   conda activate
```

### Installing and Running the Backend

A step-by-step series of examples that tell you how to get the backend server running:

1. Navigate to the `backend` directory:

   ```bash
   cd backend
   ```

2. (Optional) Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:

   ```bash
   cd app
   uvicorn main:app --reload
   ```

   The `--reload` flag enables hot reloading during development.

### Installing and Running the Frontend

Instructions to get the frontend up and running(use a new terminal, make sure the backend is running simultaneously):

1. Navigate to the `frontend` directory:

   ```bash
   cd frontend
   ```

2. Install the required npm packages:

   ```bash
   npm install
   ```

3. Start the React development server:

   ```bash
   npm start
   ```

   This will open your default web browser to `http://localhost:3000`.

## Using the Application

Click on "Choose File" button to upload the CSV for prediction.

Click on "Make Prediction" button to display the result.

## Built With

- [React](https://reactjs.org/) - The web framework used for the frontend
- [FastAPI](https://fastapi.tiangolo.com/) - The web framework used for the backend
- [Uvicorn](https://www.uvicorn.org/) - ASGI server for FastAPI


## Authors

- **Jing Ye** - *Principal Contributor* - [GitHub](https://github.com/yej002)
- **Ziyue Wang** - *Initial work* - [GitHub](https://github.com/ziywang50)
- **Zhijing Tan** - *Initial work* - [GitHub](https://github.com/zhijingt)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
