1.Demand_Forecasting_Project_Files:
      The main.py file is used to run the model, i.e. data insertion, feature building, model training, and graphical observations. AI model used is LSTM(Long Short Term Memory). 
      The error percentage of the LSTM model is ~10%.(It can vary with each training)
      Also the data required should be sequential, can be changed according to the need.
      To run main.py in the terminal write:
      
      python main.py
      


2. src folder:
      This folder contains the code for the ReactJS for the login page and Register page. For running the node I used the following command in the VSCODE terminal:
      
      1. Prerequisites
         
      Ensure you have Node.js (LTS version) installed on your system:

      Check version:

          node -v

      Check npm:
   
          npm -v
   
      
      2. Installation Commands
         
      Open your terminal in the project root directory and run the following:
      
            npm install
      
      What it does: This reads the package.json file and downloads all the necessary libraries (React, Vite, etc.) into a folder called node_modules. You only need to run this once when you first download the project.
      
      3. Running the Application
   
      Open your terminal in the project and run the following:
   
            npm run dev
            
      What it does: Starts the local development server.
      It compiles your code in real-time.
      It provides a local URL.
      Any changes you save in VS Code will automatically update the browser without a refresh (Hot Module Replacement).
      
      4. Other Useful Commands
         
      You can run the following commands too:
         
            npm run build
      
      What it does: Prepares your app for "Production." It shrinks the code, removes comments, and bundles everything into a folder named /dist. These are the files you actually upload to a real website server.
      
            npm run preview
            
      What it does: Lets you view the "Production" version of your site locally to make sure everything works perfectly before you go live.
      
      Project Structure:
   
            src/components/ - Contains the UI pieces like Login.jsx and Register.jsx.
   
            src/App.jsx - The main controller that switches between pages.
   
            src/App.css - The master styling sheet.
