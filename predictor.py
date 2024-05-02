import streamlit as st
#import requests
import pandas as pd
from utilities import predict
from nltk.tokenize import sent_tokenize

st.set_page_config(layout="wide")

def creds_entered() :
    if st.session_state["user"].strip() == "admin" and st.session_state["passwd"].strip() == "admin":
        st.session_state["authenticated"] = True
    else:
        st.session_state["authenticated"] = False
        if not st.session_state["passwd"]:
            st.warning("Please enter password.")
        elif not st.session_state["user"]:
            st.warning("Please enter username.")
        else:
            st.error("Invalid Username/Password 🤨")

def authenticate_user():
    if "authenticated" not in st.session_state or st.session_state["authenticated"] == False:
        st.subheader("Login")
        st.text_input(label="Username :", value="", key="user", on_change=creds_entered)
        st.text_input(label="Password :", value="", key="passwd", type="password", on_change=creds_entered)
        login_button = st.button("Login")
        if login_button:
            creds_entered()
        return False
    else:
        return True
        

# def predict(context, claim):
#     url = 'http://127.0.0.1:8080/predict'
#     data = {
#         'context': context,
#         'claim': claim
#     }
    
#     try:
#         response = requests.post(url, json=data)
        
#         if response.status_code == 200:
#             return response.json()
#         else:
#             return {'error': 'Failed to get prediction'}
#     except Exception as e:
#         return {'error': str(e)}
    
def result_form(result):
    if 'error' in result:
        st.error(result['error'])
    else:
        st.subheader('Label probabilities:')
        labels = ['NEI', 'REFUTED', 'SUPPORTED']
        probabilities = [result['probabilities'].get(label, 0) for label in labels]
        df = pd.DataFrame({label: [probabilities[i]] for i, label in enumerate(labels)})
        
        # Function to apply background color to cells based on label
        def apply_background(val, label):
            color = ''
            if label == 'NEI':
                color = '#FFD700'
            elif label == 'REFUTED':
                color = '#DC143C'
            else:  # Supported
                color = '#7FFF00'
            return f'background-color: {color}; color: black'
        # Apply the styling
        df_styled = df.style.apply(lambda x: [apply_background(x[name], name) for name in df.columns], axis=1)
        df_styled = df_styled.format("{:.2%}")  # Format percentages
        
        # Display the styled DataFrame
        
        st.dataframe(df_styled, hide_index=True,  use_container_width=True)
        #st.markdown(f"**Evidence:** {result['evidence'][0]}")

def get_sentences(context):
    sentences = sent_tokenize(context)
    return sentences

def create_expander_with_check_button(title, context, predict_func):
    claim_key = f"{title.upper()}_claim_entered"
    evidence_key = f"{title.upper()}_evidence_selected"
    claim_input_key = f'{title}_input'
    with st.expander(title):
        claim = st.text_input(f'Claim {title.upper()}', max_chars=500, key=claim_input_key)
        if claim:
            result = predict_func(context, claim)
            result_form(result)
            
            # Update session state variable when claim is entered
            st.session_state[claim_key] = True
            if result and 'evidence' in result:
                # Get sentences from context
                sentences = get_sentences(context)
                
                # Display sentences for evidence selection
                st.multiselect("Select evidence:", sentences, key=evidence_key)
                
        else:
            st.warning("Please enter a claim.")
            
            # Reset session state variables when claim is not entered
            st.session_state[claim_key]= ''



if 'annotated_data' not in st.session_state:
    st.session_state['annotated_data'] = pd.DataFrame(columns=['Username', 'Context', 'Claim', 'Label', 'Evidence', 'Title', 'Link'])

annotated_data = st.session_state['annotated_data']



def save_data(context, default_title, default_link):
    # Lấy DataFrame từ session state
    annotated_data = st.session_state['annotated_data']
    
    # Iterate over the claims and save them to the DataFrame
    for label in ['NEI', 'REFUTED', 'SUPPORTED']:
        claim_key = f"{label}_input"
        evidence_key = f"{label}_evidence_selected"
        
        # Check if the claim is entered
        if st.session_state.get(claim_key, ''):
            claim = st.session_state[claim_key]
            evidence = st.session_state.get(evidence_key, [])
            
            # Append data to the DataFrame
            annotated_data.loc[len(annotated_data)] = ['admin', context, claim, label, evidence, default_title, default_link]
    
    # Lưu DataFrame vào session state
    st.session_state['annotated_data'] = annotated_data



  

def predictor_app():
    if authenticate_user():
        tab1, tab2 = st.tabs(["Annotate", "Save"])
        st.sidebar.title("Dataset Upload")
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is None:
            st.sidebar.warning("Please upload a CSV file.")
        else:
            df = pd.read_csv(uploaded_file)
            max_index = len(df) - 1
            current_index = st.session_state.get("current_index", 0)
            current_row = df.iloc[current_index]
        
         
            default_context = current_row['Summary']
            default_ID = current_row['ID']
            default_title = current_row['Title']
            default_link = current_row['URL']

            with tab1:       
                st.title("Fact Checking annotation app")
                c1 = st.container(border=True)
                with c1:
                    ten_file, id_cau, chu_de, link = st.columns(4)
                    with ten_file:
                        st.text_input("Tên File:",value=uploaded_file.name)
                    with id_cau:
                        st.text_input("ID Context: ",value=default_ID)
                    with chu_de:
                        st.text_input("Chủ đề:", value=default_title)
                    with link:
                        st.text_input("Link:",value=default_link)
                    
                c2 = st.container(border=True)
                with c2:
                    nv, ev = st.columns(2)
                    with nv:
                        st.title("Nhiệm vụ")
                        st.write("Đây là nhiệm vụ tạo dữ liệu Fact Checking, với đoạn Context cho trước: annotater nhấn để đặt câu cho vô Câu Claim đặt câu cho mỗi nhãn suy luận, lần lượt với 3 nhãn Supports, Refutes và NEI (Not Enough Information). Mỗi đoạn context phải đặt ít nhất 5 câu với mỗi loại claim", height=100)
                    # with ev:
                    #     st.title("Info")
                    #     st.write(f"Username: {st.session_state.get('username', '')}")
                    #     st.write("Bank_account:  111111111111")
                
                c3 = st.container(border=True)
                with c3:
                    left_column, right_column = st.columns([0.45, 0.55])
                    with left_column:
                        st.title("Context")
                        c3_1 = st.container(border=True, height = 770)
                        with c3_1:
                            st.write(f'{default_context}')
            
                    with right_column:
                        st.title("Claim")
                        c3_2 = st.container(border=True, height = 650)
                        with c3_2:
                            # Sử dụng hàm để tạo các expander với nút kiểm tra tương ứng
                            create_expander_with_check_button("NEI", default_context, predict)
                            create_expander_with_check_button("REFUTED", default_context, predict)
                            create_expander_with_check_button("SUPPORTED", default_context, predict)
                    
                        # Check if all claims are entered
                        all_claims_entered = st.session_state.get("NEI_claim_entered", False) and \
                                              st.session_state.get("REFUTED_claim_entered", False) and \
                                              st.session_state.get("SUPPORTED_claim_entered", False)
                        
                        all_evidence_selected = (st.session_state.get("NEI_evidence_selected", []) and  \
                                 st.session_state.get("REFUTED_evidence_selected", []) and \
                                 st.session_state.get("SUPPORTED_evidence_selected", []))
                        
                        previous, next_, save, close = st.columns(4)
                        error = ''
                        with previous:
                            pr = st.button("Previous")
                            if pr and all_claims_entered:
                                if current_index > 0:
                                    st.session_state["current_index"] = current_index - 1
                                    st.experimental_rerun()
                                else:
                                    st.session_state["current_index"] = max_index
                                    st.experimental_rerun()
                            elif pr and not all_claims_entered:
                                error = 'navigate'
                        
                        with next_:
                            next_b = st.button("Next")
                            if next_b and all_claims_entered:
                                if current_index < max_index:
                                    st.session_state["current_index"] = current_index + 1
                                    st.experimental_rerun()
                                else:
                                    st.session_state["current_index"] = 0
                                    st.experimental_rerun()
                            elif next_b and not all_claims_entered:
                                error = 'navigate'
                                
        
        
                        with save:
                            save_button = st.button("Save")
                            if save_button:
                                # Check if all claims are entered before saving
                                if all_claims_entered and all_evidence_selected:
                                    # Save data
                                    save_data(default_context, default_title, default_link)
                                    error = 'success'
                                    
                                else:
                                    error = 'save_fail'
                                    
        
                        with close:
                            cl = st.button("Close")
                            if cl:
                                st.session_state["authenticated"] = False
                                st.experimental_rerun()
        
                        if error == 'navigate':
                            st.warning("Please enter all claims and select all evidence before navigating.")
                        elif error == 'success':
                             st.success("Data saved successfully.")
                        else:
                            st.warning("Please enter all claims and select all evidence before saving.")
                with tab2:
                    st.title("Saved Annotations")
                    if annotated_data.empty:
                        st.info("No annotations saved yet.")
                    else:
                        st.dataframe(annotated_data)
if __name__ == '__main__':
    predictor_app()
