

# Project Configuration
 
## Adding Credentials
 
To add credentials for the `gemini_api_key`, follow these steps:
 
1. Open the `config.json` file located in the project's root directory.
2. Add the `gemini_api_key` in the following format:
   ```json
   {
       "gemini_api_key": "YOUR_API_KEY_HERE"
   }
   ```
3. Save the `config.json` file.

## Creating Instructions.docx

goal: 
tools to be used: multiply_numbers, divide_numbers
sequence of tools:
1. multiply_numbers_1
2. divide_numbers

additional information: instruction content:

**user input/goal:** perform a calculation and send the result via email.  the input may involve multiplying numbers.  multiple email recipients may be specified.

**tool usage:**

* **`multiply_numbers`:**  this tool performs multiplication operations on numerical inputs.  its output is the result of the calculation.

* **`divide_numbers`:** this tool performs division operations on numerical inputs.  its output is the result of the calculation.

**tool sequence:**

the typical sequence involves first using `multiply_numbers` to obtain the calculation result. this result is then passed as input to `divide_numbers` to generate the result.

 
## Installing Dependencies
 
Before running the backend, make sure to install all the necessary dependencies. Use the following command to install the required packages:
 
```bash
pip install -r requirements.txt
```
 
## Running the Backend
 
To run the backend using the `main_refactorII_3_55_convotoolresultverifier.py` script, use the following command:
 
```bash
python run main_refactorII_3_55_convotoolresultverifier.py
```
 
Make sure you have all the necessary dependencies installed before running the script.

## License

The source code for the project is licensed under the MIT license, which you can find in the LICENSE.md file.

## Contact

If you have more questions or need further insights, feel free to Connect with us @ agentpro@infosys.com