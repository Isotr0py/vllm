from huggingface_hub import hf_hub_download

from vllm import LLM, SamplingParams


def run_gguf_inference(model_path, tokenizer):
    # Sample prompts.
    prompts = [
        "How many helicopters can a human eat in one sitting?",
        "What's the future of AI?",
    ]
    prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=128)

    # Create an LLM.
    llm = LLM(model=model_path, tokenizer=tokenizer)

    outputs = llm.chat(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    repo_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    filename = "qwen2.5-0.5b-instruct-q5_k_m.gguf"
    tokenizer = "Qwen/Qwen2.5-0.5B-Instruct"
    model = hf_hub_download(repo_id, filename=filename)
    run_gguf_inference(model, tokenizer)
