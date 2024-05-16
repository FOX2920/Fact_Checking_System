import streamlit as st
import pandas as pd
from utilities import predict

st.set_page_config(layout="wide")

def result_form(result, label):
    if 'error' in result:
        st.error(result['error'])
    else:
        st.subheader('Label probabilities:')
        labels = ['SUPPORTED', 'REFUTED', 'NEI']
        probabilities = {lbl: result['probabilities'].get(lbl, 0) for lbl in labels}
        
        df = pd.DataFrame({label: [probabilities[label]] for label in labels})
        
        def apply_background(val, label):
            color = ''
            if label == 'NEI':
                color = '#FFD700'
            elif label == 'REFUTED':
                color = '#DC143C'
            else:  # Supported
                color = '#7FFF00'
            return f'background-color: {color}; color: black'
        
        df_styled = df.style.apply(lambda x: [apply_background(x[name], name) for name in df.columns], axis=1)
        df_styled = df_styled.format("{:.2%}")
        
        st.dataframe(df_styled, hide_index=True, use_container_width=True)
        
        return probabilities[label] < 0.33


def create_expander_with_check_button(label, title, context, predict_func):
    claim_key = f"{label}_input"
    evidence_key = f"{label}_evidence_selected"
    label_e_ops = f"{label}_options"
    evidence_input_key = f"{label}_evidence_input"

    if label_e_ops not in st.session_state:
        st.session_state[label_e_ops] = []

    annotated_data = st.session_state['annotated_data']
    with st.expander(label, expanded=True):
        claim = st.text_input(f'Claim {label.upper()}', max_chars=500, key=claim_key)
        if claim:
            if not annotated_data[((annotated_data['Claim'] == claim) & (annotated_data['Label'] == label) & (annotated_data['Title'] == title))].empty:
                st.warning(f"This claim with label '{label}' and title '{title}' already exists.")
            else:
                result = predict_func(context, claim)
                if result_form(result, label):
                    evidence = st.text_input("Enter evidence to be added", key=evidence_input_key)
                    if evidence:
                        if evidence in context:
                            if evidence not in st.session_state[label_e_ops]:
                                st.session_state[label_e_ops].append(evidence)
                        else:
                            st.warning("Entered evidence does not appear in the context.")

                    st.multiselect(f"Select evidence for {label}", st.session_state[label_e_ops], default=st.session_state[label_e_ops], key=evidence_key)
                else:
                    st.warning(f"The predicted probability for label '{label}' is too high. Please modify the claim.")
        else:
            st.warning("Please enter a claim.")


if 'annotated_data' not in st.session_state:
    st.session_state['annotated_data'] = pd.DataFrame(columns=['Username', 'Context', 'Claim', 'Label', 'Evidence', 'Title', 'Link'])

annotated_data = st.session_state['annotated_data']

def save_data(context, default_title, default_link):
    username = st.session_state.get("username", "admin")
    annotated_data = st.session_state['annotated_data']
    error = 'success'
    
    for label in ['NEI', 'REFUTED', 'SUPPORTED']:
        claim_key = f"{label}_input"
        evidence_key = f"{label}_evidence_selected"
        
        if st.session_state.get(claim_key, ''):
            claim = st.session_state[claim_key]
            evidence = st.session_state.get(evidence_key, [])
            if not annotated_data[((annotated_data['Claim'] == claim) & (annotated_data['Label'] == label) & (annotated_data['Title'] == default_title))].empty:
                error = 'duplicate'
            else:
                annotated_data.loc[len(annotated_data)] = [username, context, claim, label, evidence, default_title, default_link]
    
    st.session_state['annotated_data'] = annotated_data
    return error

def enough_claims_entered(title):
    annotated_data = st.session_state['annotated_data']
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
                Nhiệm vụ của bạn là tạo các câu nhận định cho các nhãn sau: <span style='color:#7FFF00'>SUPPORTED</span> (Được hỗ trợ), <span style='color:#DC143C'>REFUTED</span> (Bị phủ nhận) hoặc <span style='color:#FFD700'>NEI</span> (Không đủ thông tin) dựa trên đoạn văn bản được cung cấp trước đó. Dưới đây là các bước để thực hiện nhiệm vụ này:
                
                1. **Đọc đoạn văn bản (context)**: hiểu nội dung, thông tin của đoạn văn bản được cung cấp.
                
                2. **Nhập câu nhận định**: dựa trên thông tin, nội dung đó, bạn hãy viết câu nhận định cho đoạn văn đó.
                
                3. **Phân loại câu nhận định**: sau khi đã viết xong câu nhận định, bạn hãy sắp xếp nó vào một trong ba nhãn sau:
                    - <span style='color:#7FFF00'>SUPPORTED</span> (được hỗ trợ): đây là nhãn mà khi câu nhận định của bạn là chính xác theo những thông tin nội dung của đoạn văn bản (context) cung cấp.
                    - <span style='color:#DC143C'>REFUTED</span> (bị bác bỏ): ngược lại với “<span style='color:#7FFF00'>SUPPORTED</span>”, đây là nhãn mà khi câu nhận định của bạn là sai so với những thông tin nội dung của đoạn văn bản (context) đưa ra.
                    - <span style='color:#FFD700'>NEI</span> (không đủ thông tin): khi thông tin mà câu nhận định của bạn đưa ra chưa thể xác định được đúng hoặc sai dựa trên thông tin của đoạn văn bản (context) cung cấp; hoặc ít nhất một thông tin mà bạn đưa ra trong câu nhận định không xuất hiện ở đoạn văn bản (context).
                
                4. **Chọn bằng chứng (Evidence)**: đối với hai nhãn <span style='color:#7FFF00'>SUPPORTED</span> & <span style='color:#DC143C'>REFUTED</span>, các bạn sẽ chọn bằng chứng (evidence) cho câu nhận định. Nghĩa là các bạn sẽ chọn những thông tin trong đoạn văn bản (context) để dựa theo đó để chứng minh rằng câu nhận định của bạn là đúng (đối với “<span style='color:#7FFF00'>SUPPORTED</span>”) hoặc sai (đối với “<span style='color:#DC143C'>REFUTED</span>”). Các bạn chỉ chọn những thông tin cần thiết (không chọn hết cả câu hoặc cả đoạn văn).
                
                5. **Lưu dữ liệu**: Sau khi đã nhập đủ 2 câu (mỗi nhãn một câu), bạn nhấn vào nút “Save” bên dưới để lưu lại các câu đó. Sau khi lưu hoàn tất, thông báo sẽ hiện và các câu đã viết trước đó sẽ được xóa để bạn có thể nhập câu mới.
                
                6. **Di chuyển đến đoạn văn bản (context) khác**: bạn có thể di chuyển qua lại giữa các context nhưng chỉ khi bạn đã tạo tối thiểu 6 câu nhận định (mỗi nhãn tối thiểu 2 câu).
                
                7. **Đóng ứng dụng**: Khi muốn kết thúc phiên làm việc, bạn ấn nút Close để đóng ứng dụng. Nhớ save lại (bước 5) trước khi thoát nhé! ❤️
                
                Xem chi tiết hướng dẫn cách đặt câu nhận định [tại đây](https://docs.google.com/document/d/121GHPAOFa4_fhmXDGJFYCrmsStcXYc7H/edit).
                
                Lấy các đoạn văn bản (context): [tại đây](https://drive.google.com/drive/folders/1bbW7qiglBZHvGs5oNF-s_eac09t5oWOW).
                """, unsafe_allow_html=True)  

    if uploaded_file is None:
        st.sidebar.warning("Please upload a CSV file.")
    else:
        df = pd.read_csv(uploaded_file)
        require_columns = ['Summary', 'ID', 'Title', 'URL']
        
        if not set(require_columns).issubset(df.columns):
            st.error("Error: Upload Dataset is missing require columns.")
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
                    st.text_input("Tên File:", value=uploaded_file.name, disabled=True)
                with id_cau:
                    st.text_input("ID Context: ", value=default_ID, disabled=True)
                with chu_de:
                    st.text_input("Chủ đề:", value=default_title, disabled=True)
                with link:
                    st.text_input("Link:", value=default_link, disabled=True)
                
            c3 = st.container(border=True)
            with c3:
                left_column, right_column = st.columns([0.45, 0.55])
                with left_column:
                    st.title("Context")
                    c3_1 = st.container(border=True, height=770)
                    with c3_1:
                        st.write(f'{default_context}')
        
                with right_column:
                    st.title("Claim")
                    c3_2 = st.container(border=True, height=650)
                    with c3_2:
                        create_expander_with_check_button("SUPPORTED", default_title, default_context, predict)
                        create_expander_with_check_button("REFUTED", default_title, default_context, predict)
                        create_expander_with_check_button("NEI", default_title, default_context, predict)
                
                    all_claims_entered = st.session_state.get("NEI_claim_entered", False) and \
                                         st.session_state.get("REFUTED_claim_entered", False) and \
                                         st.session_state.get("SUPPORTED_claim_entered", False)
                    
                    all_evidence_selected = (st.session_state.get("NEI_evidence_selected", []) and \
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
                            if all_claims_entered and all_evidence_selected:
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
