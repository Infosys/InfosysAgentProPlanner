# © 2024-25 Infosys Limited, Bangalore, India. All Rights Reserved.
from urllib import response
import streamlit as st
import requests
import pandas as pd 
import re 

#-- configuration
st. set_page_config(
    page_title="Agent Evaluation Interface",
    layout="wide"
)

#This should point to the address where your FastAPI backend is running
API_BASE_URL="http://127.0.0.1:6001"

#---Helper Function

def get_filename_from_header(header):
    """
    Extracts the filename from the Content-disposition header.
    Example: 'attachment; filename="evalution_result_xyz.xlsx"'->'evaluation_result_xyz.xlsx'
    """
    if not header:
        return "evaluation_result.xlsx"
    
    match=re.search(r'filename="?([^"]+)"?',header)
    if match:
        return match.group(1)
    return "evaluation_result.xlsx"



#--Main UI---
st.title("Agent Evaluation Service")
st.markdown("---")

#--- Sidebar for workflow Selection ---
st.sidebar.title("Workflows")
workflow=st.sidebar.radio("Choose an Evaluation Workflow:",("File Evaluation"))

if workflow=="File Evaluation" :
    st.header("Workflow : File Evaluation")
    st.info(
        """
This workflow processes an uploaded file **entirely in memory** and provides an immediate download.
1. **Configure** : Enter the names of the two models to use for evaluation.
2. **Upload** : Provide a CSV or Excel file containing the evaluation data
3. **Execute & Download** : The file is processes immediately. Once complete, a download button will appear with the results

"""
    )


    # <<< START : NEW SECTION FOR SAMPLE FILE DOWNLOAD >>>
    st.markdown("---")
    st.subheader("Need a template?")
    st.markdown("Downaload a sample Excel file with the required column and format.")

    try:
        # Read the local sample file into memory as bytes.
        # This assumes 'sample_Data.xlsx' is in the same folder as this script.
        with open("sample_data.xlsx","rb") as f:
            file_bytes=f.read()

        #Create the download button
        st.download_button(
            label="Download sample_data.xlsx",
            data=file_bytes,
            file_name="sample_data.xlsx",
            mime="application/vnd.openxmlformats-offiedocument.spreadsheetml.sheet",
            help="Click to download a sample file with the correct structure."
        )
    except FileNotFoundError:
        st.warning(
            " ** Sample file not found.** To enable this download, please add a file named 'sample_data.xlsx' to the same folder as the streamlit app."
        )

    st.markdown("---")
    # <<< END : NEW SECTION FOR SAMPLE FILE DOWNLOAD >>>

    st.subheader ("Configure and Upload")

    with st.form("file_eval_form"):
        col1,col2=st.columns(2)

        with col1:
            model1_file=st.text_input("Enter Model 1 Name", value="gpt-4o")
        
        with col2:
            model2_file=st.text_input("Enter Model 2 Name", value="gpt-4o-mini")

        uploaded_file= st.file_uploader(
            "Upload your evaluation file",
            type=["csv","xlsx","xls"],
            help="The file should contain columns like 'id', 'query', 'response', 'model_used', etc."
        )

        submit_button_file=st.form_submit_button(
            label=" Process File and Prepare Download",
            use_container_width=True
        )    

    if submit_button_file:
        if not model1_file or not model2_file:
            st.error("Please select both Model 1 and Model 2 names.")
        elif uploaded_file is None:
            st.error("Please upload a file to process.")
        else:
            endpoint = f"{API_BASE_URL}/v1/evaluation/process-file-and-download"

            #Prepare the multipart/form-data payload
            form_data={
                "model1":(None, model1_file),
                "model2" : (None,model2_file)

            }

            files={
                "file" :(uploaded_file.name,uploaded_file.getvalue(),uploaded_file.type)
            }

            with st.spinner(f"Processing '{uploaded_file.name}'... This may take several minutes depending on file size and model speed."):
                try:
                    response= requests.post(endpoint, data=form_data, files=files)

                    if response.status_code == 200:
                        st.success("** Processing complete!** Your download is ready.")

                        #Get filename from header and prepare download button
                        filename =get_filename_from_header(response.headers.get("Content-Disposition"))

                        st.download_button(
                            label= f"Download {filename}",
                            data=file_bytes,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-offiedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

                        st.balloons()

                    else:
                        st.error(f"**Error!** Filed to process file. Server responded with status {response.status_code}.")
                        st.json(response.json())

                except requests.exceptions.RequestException as e:
                    st.error (f"Failed to connect to the API. Please Ensure the backend is running at {API_BASE_URL}.")
                    st.error(f"Details:{e}")