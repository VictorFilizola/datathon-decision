# Datathon Decision - Project MVP

This project is a complete Machine Learning application designed to match job applicants with suitable vacancies.

## How to Run This Project

This project consists of a Python backend (API) and a Python frontend (Web Interface). Please follow these steps to run the application.

**Prerequisites:**
- Python 3.9+ installed on your machine.
- Pip (Python's package installer).

---

### Step 1: Install Dependencies

1.  Unzip the project folder.
2.  Open a command prompt or terminal and navigate into the project's root directory.
3.  Install all the required Python libraries by running the following command:
    ```bash
    pip install -r requirements.txt
    ```

---

### Step 2: Run the Application

1.  Once the installation is complete, simply double-click the `start_app.bat` file located in the project's root directory.
2.  This batch file will automatically:
    * Start the backend API server in a new command prompt window.
    * Start the frontend web interface in a second command prompt window.
    * Open the application in your default web browser.

3.  If the browser does not open automatically, you can access the application by navigating to:
    **http://localhost:8501**

---

### Step 3: Using the Test Scripts

The project includes standalone test scripts located in the `/tests` folder. To run them, navigate to the project root in your terminal and execute:

* **To test the core ML functions:**
    ```bash
    python tests/test_ml.py
    ```

* **To test all API endpoints automatically:**
    ```bash
    python tests/test_api.py
    ```