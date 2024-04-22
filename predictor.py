import streamlit as st
import requests

def predict(context, claim):
    url = 'http://127.0.0.1:8080/predict'
    data = {
        'context': context,
        'claim': claim
    }
    
    try:
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': 'Failed to get prediction'}
    except Exception as e:
        return {'error': str(e)}
    
def result_form(result):
    if 'error' in result:
        st.error(result['error'])
    else:
        st.subheader('Prediction Result:')
        st.markdown(f"**Predicted Label:** {result['predicted_label']}")
        st.markdown("**Probabilities:**")
        for label, prob in result['probabilities'].items():
            color = ''
            if label == 'NEI':
                color = '#FFD700'  # Màu vàng nhạt
            elif label == 'REFUTED':
                color = '#DC143C'  # Màu đỏ đậm
            elif label == 'SUPPORTED':
                color = '#7FFF00'  # Màu xanh lá
            st.markdown(f"- <font color='{color}'>{label}: {prob}</font>", unsafe_allow_html=True)
        st.markdown(f"**Evidence:** {result['evidence'][0]}")


def predictor_app():
    st.set_page_config(layout="wide")

    ten_file, id_cau, chu_de, link = st.columns(4)
    with ten_file:
        ten_file_input = st.text_input("Tên File:")
    with id_cau:
        id_cau_input = st.text_input("ID Câu:")
    with chu_de:
        chu_de_input = st.text_input("Chủ đề:")
    with link:
        link_input = st.text_input("Link:")
        
    c1 = st.container(border=True)
    with c1:
        nv, ev = st.columns(2)
        with nv:
            st.title("Nhiệm vụ")
            st.write("Đây là nhiệm vụ tạo dữ liệu FC, với Câu Evidence cho trước: annotater nhấn đề đã câu cho vô Câu Claim đại hai câu cho mỗi nhãn suy luận, lần lượt với 3 nhân Supports, Refutes và NEI (Not Enough Information) so với Câu Evidence. Phải điền đầy đủ cả 6 ô trước khi next sang câu mới.", height=100)
        with ev:
            st.title("Evidence")
            st.write("Evidence", height=100)
    
    default_context = "Trái Đất là hành tinh duy nhất trong Hệ Mặt Trời được biết đến là nơi có sự sống tồn tại. Nó là hành tinh lớn thứ ba trong hệ này về kích thước và khối lượng. Trái Đất hình cầu với bề mặt gồm nước và đất liền, được bao phủ bởi lớp khí quyển. Khí quyển của Trái Đất chủ yếu bao gồm nitơ và oxy, cùng với các khí nhà kính như hơi nước và carbon dioxide. Trái Đất quay quanh Mặt Trời theo một quỹ đạo hình ellip, hoàn thành một vòng quay trong khoảng 365 ngày, gây ra sự luân phiên của các mùa."
    default_claim = "Trái Đất là hành tinh duy nhất trong Hệ Mặt Trời được biết đến là nơi có sự sống tồn tại."
    c2 = st.container(border=True)
    with c2:
        left_column, right_column = st.columns([3, 2])
        with left_column:
            st.title("Context")
            context = st.text_area('Context', value=default_context, height=1050, max_chars=500)
   
        with right_column:
            with st.expander("NEI"):
                claim_nei = st.text_input('Claim NEI', value=default_claim, max_chars=500)
                check_button_nei = st.button('Check NEI')
                if check_button_nei:
                    result_nei = predict(context, claim_nei)
                    result_form(result_nei)
            
            with st.expander("REFUTED"):
                claim_refuted = st.text_input('Claim REFUTED', value=default_claim, max_chars=500)
                check_button_refuted = st.button('Check REFUTED')
                if check_button_refuted:
                    result_refuted = predict(context, claim_refuted)
                    result_form(result_refuted)
            
            with st.expander("SUPPORTED"):
                claim_supported = st.text_input('Claim SUPPORTED', value=default_claim, max_chars=500)
                check_button_supported = st.button('Check SUPPORTED')
                if check_button_supported:
                    result_supported = predict(context, claim_supported)
                    result_form(result_supported)

if __name__ == '__main__':
    predictor_app()
