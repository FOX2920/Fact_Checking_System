import streamlit as st
#import requests
import pandas as pd
from utilities import predict
from nltk.tokenize import sent_tokenize

st.set_page_config(layout="wide")

        

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

def create_expander_with_check_button(label, title, context, predict_func):
    claim_key = f"{label.upper()}_claim_entered"
    evidence_key = f"{label.upper()}_evidence_selected"
    claim_input_key = f'{label}_input'

    # Lấy DataFrame từ session state
    annotated_data = st.session_state['annotated_data']
    with st.expander(label, expanded=True):
        claim = st.text_input(f'Claim {label.upper()}', max_chars=500, key=claim_input_key)
        if claim:
            if not annotated_data[((annotated_data['Claim'] == claim) & (annotated_data['Label'] == label) & (annotated_data['Title'] == title))].empty:
                st.warning(f"This claim with label '{label}' and title '{title}' already exists.")
            else:
                result = predict_func(context, claim)
                result_form(result)
                
                # Update session state variable when claim is entered
                st.session_state[claim_key] = True
                if result and 'evidence' in result:
                    # Get sentences from context
                    sentences = get_sentences(context)
                    
                    # Display sentences for evidence selection
                    st.multiselect("Select evidence:", sentences, key=evidence_key, max_selections=5)
                
        else:
            st.warning("Please enter a claim.")
            
            # Reset session state variables when claim is not entered
            st.session_state[claim_key]= ''



if 'annotated_data' not in st.session_state:
    st.session_state['annotated_data'] = pd.DataFrame(columns=['Username', 'Context', 'Claim', 'Label', 'Evidence', 'Title', 'Link'])

annotated_data = st.session_state['annotated_data']



def save_data(context, default_title, default_link):
    # Lấy tên người dùng từ session state
    username = st.session_state.get("username", "admin")
    
    # Lấy DataFrame từ session state
    annotated_data = st.session_state['annotated_data']
    error = 'success'
    # Iterate over the claims and save them to the DataFrame
    for label in ['NEI', 'REFUTED', 'SUPPORTED']:
        claim_key = f"{label}_input"
        evidence_key = f"{label}_evidence_selected"
        
        # Check if the claim is entered
        if st.session_state.get(claim_key, ''):
            claim = st.session_state[claim_key]
            evidence = st.session_state.get(evidence_key, [])
            if not annotated_data[((annotated_data['Claim'] == claim) & (annotated_data['Label'] == label) & (annotated_data['Title'] == default_title))].empty:
                error = 'duplicate'
            else:
                # Append data to the DataFrame
                annotated_data.loc[len(annotated_data)] = [username, context, claim, label, evidence, default_title, default_link]
    
    # Lưu DataFrame vào session state
    st.session_state['annotated_data'] = annotated_data
    return error

def enough_claims_entered(title):
    # Lấy DataFrame từ session state
    annotated_data = st.session_state['annotated_data']
    
    # Kiểm tra xem đã đủ ít nhất năm claim cho mỗi nhãn với title cụ thể không
    nei_claims = annotated_data[(annotated_data['Label'] == 'NEI') & (annotated_data['Title'] == title)].shape[0]
    refuted_claims = annotated_data[(annotated_data['Label'] == 'REFUTED') & (annotated_data['Title'] == title)].shape[0]
    supported_claims = annotated_data[(annotated_data['Label'] == 'SUPPORTED') & (annotated_data['Title'] == title)].shape[0]

    return nei_claims >= 3 and refuted_claims >= 3 and supported_claims >= 3


  

def predictor_app():
    tab0, tab1, tab2 = st.tabs(["Mission", "Annotate", "Save"])
    st.sidebar.title("Dataset Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    with tab0:
        c2 = st.container(border=True)
        with c2:
                st.title("Nhiệm vụ")
                st.write("""
                    Nhiệm vụ của bạn là tạo các câu nhận định cho các nhãn sau: "SUPPORTED" (Được hỗ trợ), "REFUTED" (Bị phủ nhận) hoặc "NEI" (Không đủ thông tin) dựa trên đoạn văn bản được cung cấp trước đó. Dưới đây là các bước để thực hiện nhiệm vụ này:
                        1.	**Đọc đoạn văn bản (context)**: hiểu nội dung, thông tin của đoạn văn bản được cung cấp.
                        2.	**Nhập câu nhận định**: dựa trên thông tin, nội dung đó, bạn hãy viết câu nhận định cho đoạn văn đó. 
                        3.	**Phân loại câu nhận định**: sau khi đã viết xong câu nhận định, bạn hãy sắp xếp nó vào một trong ba nhãn sau:
                        -	“SUPPORTED” (được hỗ trợ): đây là nhãn mà khi câu nhận định của bạn là chính xác theo những thông tin nội dung của đoạn văn bản (context) cung cấp
                        -	“REFUTED” (bị bác bỏ): ngược lại với “SUPPORTED”, đây là nhãn mà khi câu nhận định của bạn là sai so với những thông tin nội dung của đoạn văn bản (context) đưa ra
                        -	“NEI” (không đủ thông tin): khi thông tin mà câu nhận định của bạn đưa ra chưa thể xác định được đúng hoặc sai dựa trên thông tin của đoạn văn bản (context) cung cấp; hoặc ít nhất một thông tin mà bạn đưa ra trong câu nhận định không xuất hiện ở đoạn văn bản (context)
                        ** Lưu ý rằng các bạn sẽ không sử dụng các kiến thức, thông tin ở bên ngoài mà chỉ dựa vào nội dung đoạn văn bản (context) cung cấp để đưa ra câu nhận định và sắp xếp nó vào nhãn thích hợp
                        4.	**Chọn bằng chứng (Evidence)**: đối với hai nhãn “SUPPORTED” & “REFUTED”, các bạn sẽ chọn bằng chứng (evidence) cho câu nhận định. Nghĩa là các bạn sẽ chọn những thông tin trong đoạn văn bản (context) để dựa theo đó để chứng minh rằng câu nhận định của bạn là đúng (đối với “SUPPORTED”) hoặc sai (đối với “REFUTED”). Các bạn chỉ chọn những thông tin cần thiết (không chọn hết cả câu hoặc cả đoạn văn)
                        5.	**Lưu dữ liệu**: Sau khi đã nhập đủ 2 câu (mỗi nhãn một câu), bạn nhấn vào nút “Save” bên dưới để lưu lại các câu đó. Sau khi lưu hoàn tất, thông báo sẽ hiện và các câu đã viết trước đó sẽ được clear để viết câu mới.
                        6.	**Di chuyển đến đoạn văn bản (context) khác**: bạn có thể di chuyển qua lại giữa các context với nhau nhưng chỉ khi bạn đã tạo tối thiểu 6 câu nhận định (mỗi nhãn tối thiểu 2 câu)
                        7.	**Đóng ứng dụng**: khi muốn kết thúc phiên làm việc, bạn ấn nút “close” để đóng ứng dụng ---- nhớ save lại (bước 5) trước khi thoát nhé <3
                        
                        Xem chi tiết hướng dẫn cách đặt câu nhận định: (tại đây)[https://docs.google.com/document/d/121GHPAOFa4_fhmXDGJFYCrmsStcXYc7H/edit].
                        Lấy các đoạn văn bản (context): (tại đây)[https://drive.google.com/drive/folders/1bbW7qiglBZHvGs5oNF-s_eac09t5oWOW].
            """)
    if uploaded_file is None:
        st.sidebar.warning("Please upload a CSV file.")
    else:
        df = pd.read_csv(uploaded_file)
        required_columns = ['Summary', 'ID', 'Title', 'URL']  # Các cột cần thiết
        
        # Kiểm tra xem tất cả các cột cần thiết có tồn tại không
        if not set(required_columns).issubset(df.columns):
            st.error("Error: Upload Dataset is missing required columns.")
            st.stop()
        else:
            max_index = len(df) - 1
            current_index = st.session_state.get("current_index", 0)
            current_row = df.iloc[current_index]
            
            default_context = current_row['Summary']
            default_ID = current_row['ID']
            default_title = current_row['Title']
            default_link = current_row['URL']
    with tab1:
        if uploaded_file is None:
            st.error("Dataset not found")
        else:
            st.title("Fact Checking annotation app")
            c1 = st.container(border=True)
            with c1:
                ten_file, id_cau, chu_de, link = st.columns(4)
                with ten_file:
                    st.text_input("Tên File:",value=uploaded_file.name, disabled=True)
                with id_cau:
                    st.text_input("ID Context: ",value=default_ID, disabled=True)
                with chu_de:
                    st.text_input("Chủ đề:", value=default_title, disabled=True)
                with link:
                    st.text_input("Link:",value=default_link, disabled=True)
                
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
                        create_expander_with_check_button("NEI", default_title, default_context, predict)
                        create_expander_with_check_button("REFUTED", default_title, default_context, predict)
                        create_expander_with_check_button("SUPPORTED", default_title, default_context, predict)
                
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
                        if pr:
                            if enough_claims_entered(default_title):
                                if current_index > 0:
                                    st.session_state["current_index"] = current_index - 1
                                    st.experimental_rerun()
                                else:
                                    st.session_state["current_index"] = max_index
                                    st.experimental_rerun()
                            else:
                                error = 'n_enough'
                    
                    with next_:
                        next_b = st.button("Next")
                        if next_b:
                            if enough_claims_entered(default_title):
                                if current_index < max_index:
                                    st.session_state["current_index"] = current_index + 1
                                    st.experimental_rerun()
                                else:
                                    st.session_state["current_index"] = 0
                                    st.experimental_rerun()
                            else:
                                error = 'n_enough'
                            
    
    
                    with save:
                        save_button = st.button("Save")
                        if save_button:
                            # Check if all claims are entered before saving
                            if all_claims_entered and all_evidence_selected:
                                # Save data
                                error = save_data(default_context, default_title, default_link)
                            else:
                                error = 'save_fail'
                                
    
                    with close:
                        cl = st.button("Close")
                        if cl:
                            st.session_state["authenticated"] = False
                            st.experimental_rerun()
    
                    if error == 'success':
                         st.success("Data saved successfully.")
                    elif error == 'duplicate':
                        st.warning(f"Maybe one of these claims with title '{default_title}' already exists.")
                    elif error == 'n_enough':
                         st.warning("Enter at least three claims for each label for this title before navigating.")
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
