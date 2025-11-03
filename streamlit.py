import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig



st.title('Grammarly for Hindi')
st.write('Enter a hindi sentence:')
input = st.text_input("Enter sentence here")


def generate_output():
    
    config = PeftConfig.from_pretrained("finetuned-hindi-model-8k")
    base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base, "finetuned-hindi-model-8k")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    tokenizer.pad_token = tokenizer.eos_token

    merged_model = model.merge_and_unload()
    messages = [
    {"role": "system", "content": "You are a hindi language expert in error correction. Correct any type of hindi error you see in the given line and return the corrected version as output."},
    {"role": "user", "content": f"Correct grammar for: {input}"},
    ]

    with st.spinner("Processing, please wait"):
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True # Important for starting new conversations
        ).to(model.device)

        generated_ids = merged_model.generate(
        input_ids=input_ids,
        max_new_tokens=512, # Adjust as needed
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.1,
        )
        output_text = tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return output_text


if st.button("Generate correct version"):
    sentence = generate_output()
    sentence = sentence.replaceAll('</s>', "")
    st.write(f"Corrected version: {sentence}")

