#!/usr/bin/env python3

import argparse
import pickle

import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import faster_whisper

import torch

import io
import datasets
import jiwer
import json
import pandas as pd

import soundfile
import numpy
import pydub
import pandas

import whisper
import whisper.normalizers

# Supported engines and models:
#
# 1. Engine: openai-whisper
#    - Models: large-v2, large-v3
# 2. Engine: transformers
#    - Models: openai/whisper-large-v2, openai/whisper-large-v3, user-trained models
# 3. Engine: faster-whisper
#    - Models: large-v2, large-v3, user-trained models


def initialize_model(engine, model_path, tuned_model_path):
    if engine == "openai-whisper":
        model = whisper.load_model(model_path)

        def transcribe(entry):
            return transcribe_openai_whisper(model, entry)

        return transcribe

    if engine == "openai-whisper-tuned":
        model = whisper.load_model(model_path)

        if tuned_model_path:
            print(f"Loading tuned model {tuned_model_path}...")
            tuned_model = WhisperForConditionalGeneration.from_pretrained(tuned_model_path)
            model = copy_hf_model_weights_to_openai_model_weights(tuned_model, model)

        def transcribe(entry):
            return transcribe_openai_whisper(model, entry)

        return transcribe

    if engine == "transformers":
        model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32)
        model.to("cuda:0")

        processor = WhisperProcessor.from_pretrained(model_path)

        def transcribe(entry):
            return transcribe_transformers(model, processor, entry)

        return transcribe

    if engine == "faster-whisper":
        model = faster_whisper.WhisperModel(model_path, device="cuda")

        def transcribe(entry):
            return transcribe_faster_whisper(model, entry)

        return transcribe

    raise Exception


def transcribe_transformers(model, processor, entry):
    audio_resample = librosa.resample(entry["audio"]["array"], orig_sr=entry["audio"]["sampling_rate"], target_sr=16000)
    input_features = processor(audio_resample, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to("cuda:0")
    predicted_ids = model.generate(input_features, language="he", num_beams=5)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]


def transcribe_faster_whisper(model, entry):
    wav_buffer = io.BytesIO()
    soundfile.write(wav_buffer, entry["audio"]["array"], entry["audio"]["sampling_rate"], format="WAV")
    wav_buffer.seek(0)

    texts = []
    segs, dummy = model.transcribe(wav_buffer, language="he")
    for s in segs:
        texts.append(s.text)

    return " ".join(texts)


def copy_hf_model_weights_to_openai_model_weights(tuned_model, model):
    dic_parameter_mapping = dict()

    for i in range(32):
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.key.weight"] = (
            f"model.decoder.layers.{i}.self_attn.k_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.out.bias"] = f"model.decoder.layers.{i}.self_attn.out_proj.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.out.weight"] = (
            f"model.decoder.layers.{i}.self_attn.out_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.query.bias"] = f"model.decoder.layers.{i}.self_attn.q_proj.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.query.weight"] = (
            f"model.decoder.layers.{i}.self_attn.q_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.value.bias"] = f"model.decoder.layers.{i}.self_attn.v_proj.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.value.weight"] = (
            f"model.decoder.layers.{i}.self_attn.v_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.attn_ln.bias"] = (
            f"model.decoder.layers.{i}.self_attn_layer_norm.bias"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.attn_ln.weight"] = (
            f"model.decoder.layers.{i}.self_attn_layer_norm.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.key.weight"] = (
            f"model.decoder.layers.{i}.encoder_attn.k_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.out.bias"] = (
            f"model.decoder.layers.{i}.encoder_attn.out_proj.bias"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.out.weight"] = (
            f"model.decoder.layers.{i}.encoder_attn.out_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.query.bias"] = (
            f"model.decoder.layers.{i}.encoder_attn.q_proj.bias"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.query.weight"] = (
            f"model.decoder.layers.{i}.encoder_attn.q_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.value.bias"] = (
            f"model.decoder.layers.{i}.encoder_attn.v_proj.bias"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.value.weight"] = (
            f"model.decoder.layers.{i}.encoder_attn.v_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn_ln.bias"] = (
            f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn_ln.weight"] = (
            f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp.0.bias"] = f"model.decoder.layers.{i}.fc1.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp.0.weight"] = f"model.decoder.layers.{i}.fc1.weight"
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp.2.bias"] = f"model.decoder.layers.{i}.fc2.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp.2.weight"] = f"model.decoder.layers.{i}.fc2.weight"
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp_ln.bias"] = f"model.decoder.layers.{i}.final_layer_norm.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp_ln.weight"] = f"model.decoder.layers.{i}.final_layer_norm.weight"

        dic_parameter_mapping[f"encoder.blocks.{i}.attn.key.weight"] = (
            f"model.encoder.layers.{i}.self_attn.k_proj.weight"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.out.weight"] = (
            f"model.encoder.layers.{i}.self_attn.out_proj.weight"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.query.weight"] = (
            f"model.encoder.layers.{i}.self_attn.q_proj.weight"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.value.weight"] = (
            f"model.encoder.layers.{i}.self_attn.v_proj.weight"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp.2.weight"] = f"model.encoder.layers.{i}.fc2.weight"
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.out.bias"] = f"model.encoder.layers.{i}.self_attn.out_proj.bias"
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.query.bias"] = f"model.encoder.layers.{i}.self_attn.q_proj.bias"
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.value.bias"] = f"model.encoder.layers.{i}.self_attn.v_proj.bias"
        dic_parameter_mapping[f"encoder.blocks.{i}.attn_ln.bias"] = (
            f"model.encoder.layers.{i}.self_attn_layer_norm.bias"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.attn_ln.weight"] = (
            f"model.encoder.layers.{i}.self_attn_layer_norm.weight"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp.2.bias"] = f"model.encoder.layers.{i}.fc2.bias"
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp_ln.bias"] = f"model.encoder.layers.{i}.final_layer_norm.bias"
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp_ln.weight"] = f"model.encoder.layers.{i}.final_layer_norm.weight"
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp.0.weight"] = f"model.encoder.layers.{i}.fc1.weight"
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp.0.bias"] = f"model.encoder.layers.{i}.fc1.bias"

    dic_parameter_mapping["encoder.conv1.bias"] = "model.encoder.conv1.bias"
    dic_parameter_mapping["encoder.conv1.weight"] = "model.encoder.conv1.weight"
    dic_parameter_mapping["encoder.conv2.bias"] = "model.encoder.conv2.bias"
    dic_parameter_mapping["encoder.conv2.weight"] = "model.encoder.conv2.weight"
    dic_parameter_mapping["encoder.ln_post.bias"] = "model.encoder.layer_norm.bias"
    dic_parameter_mapping["encoder.ln_post.weight"] = "model.encoder.layer_norm.weight"
    dic_parameter_mapping["encoder.positional_embedding"] = "model.encoder.embed_positions.weight"

    dic_parameter_mapping["decoder.ln.bias"] = "model.decoder.layer_norm.bias"
    dic_parameter_mapping["decoder.ln.weight"] = "model.decoder.layer_norm.weight"
    dic_parameter_mapping["decoder.positional_embedding"] = "model.decoder.embed_positions.weight"
    dic_parameter_mapping["decoder.token_embedding.weight"] = "model.decoder.embed_tokens.weight"

    model_state_dict = model.state_dict()
    tuned_model_state_dict = tuned_model.state_dict()

    for source_param, target_param in dic_parameter_mapping.items():
        model_state_dict[source_param] = tuned_model_state_dict[target_param]

    model.load_state_dict(model_state_dict)

    return model


def transcribe_openai_whisper(model, entry):
    wav_buffer = io.BytesIO()
    soundfile.write(wav_buffer, entry["audio"]["array"], entry["audio"]["sampling_rate"], format="WAV")
    wav_buffer.seek(0)  # Rewind to the start of the buffer

    audio = pydub.AudioSegment.from_file(wav_buffer, format="wav")
    audio.export("x.mp3", format="mp3")

    return model.transcribe("x.mp3", language="he", beam_size=5, best_of=5)["text"]


def evaluate_model(transcribe_fn, ds, text_column):
    normalizer = whisper.normalizers.BasicTextNormalizer()

    ref_texts = []
    eval_texts = []
    
    entries_data = []

    for i in range(len(ds)):
        ref_text = normalizer(ds[i][text_column])
        eval_text = normalizer(transcribe_fn(ds[i]))

        ref_texts.append(ref_text)
        eval_texts.append(eval_text)

        entry_metrics  = jiwer.process_words([ref_text], [eval_text])

        entry_data = {
            'id': i,
            'reference_text': ref_text,
            'predicted_text': eval_text,
            'wer': entry_metrics.wer,
            'wil': entry_metrics.wil,
            'substitutions': entry_metrics.substitutions,
            'deletions': entry_metrics.deletions,
            'insertions': entry_metrics.insertions,
            'hits': entry_metrics.hits,
        }
        for key in ds[i].keys():
            if key not in ['audio', text_column]:  # Skip audio array and text we already processed
                entry_data[f'metadata_{key}'] = ds[i][key]
        
        entries_data.append(entry_data)
        print(f"Evaluated {i+1}/{len(ds)} entries, WER={entry_metrics.wer}, WIL={entry_metrics.wil}")

    final_metrics = jiwer.process_words(
        [entry['reference_text'] for entry in entries_data],
        [entry['predicted_text'] for entry in entries_data]
    )

    results_df = pd.DataFrame(entries_data)
    
    return final_metrics, results_df

if __name__ == "__main__":
    # Define an argument parser
    parser = argparse.ArgumentParser(description="Create a dataset and upload to Huggingface.")

    # Add the arguments
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        choices={"openai-whisper", "openai-whisper-tuned", "transformers", "faster-whisper"},
        help="Engine to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use. Can be remote (e.g. openai/whisper-large-v3) or local (full path).",
    )
    parser.add_argument("--tuned-model", type=str, required=False)
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to evaluate in format dataset_name:<split>:<text_column>.")
    parser.add_argument("--name", type=str, required=False, help="Optional name parameter for dataset.load_dataset.")
    parser.add_argument("--output", type=str, required=False, default="evaluation_results.csv", help="Output CSV file path. If not provided, will use 'evaluation_results.csv'")
    # Parse the arguments
    args = parser.parse_args()

    print(f"Loading engine {args.engine} with model {args.model}...")
    transcribe_fn = initialize_model(args.engine, args.model, args.tuned_model)

    print(f"Loading dataset {args.dataset}...")
    dataset_parts = args.dataset.split(':')
    dataset_name = dataset_parts[0]
    dataset_split = dataset_parts[1] if len(dataset_parts) > 1 else "test"
    ds_text_column = dataset_parts[2] if len(dataset_parts) > 2 else "text"

    if args.name:
        ds = datasets.load_dataset(dataset_name, name=args.name)[dataset_split]
    else:
        ds = datasets.load_dataset(dataset_name)[dataset_split]

    print(f"Beginning evaluation.")
    metrics, results_df = evaluate_model(transcribe_fn, ds, ds_text_column)

    print(f"Evaluation done. WER={metrics.wer}, WIL={metrics.wil}.")
    
    # Add model and dataset info as columns 
    results_df['model'] = args.model
    results_df['dataset'] = dataset_name
    results_df['dataset_split'] = dataset_split
    results_df['engine'] = args.engine
    
    output_file = args.output 
    
    results_df.to_csv(output_file, encoding='utf-8', index=False)
    
    print(f"Results saved to {output_file}")
    
