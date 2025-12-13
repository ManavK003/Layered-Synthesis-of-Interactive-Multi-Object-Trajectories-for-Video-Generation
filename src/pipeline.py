import argparse
import sys
from pathlib import Path

def run_data_preparation(source_dir, output_dir, hours):
    print("=" * 60)
    print("STEP 1: Preparing Animal Kingdom Subset")
    print("=" * 60)
    
    from prepare_animal_kingdom_subset import AnimalKingdomSubsetPreparer
    
    preparer = AnimalKingdomSubsetPreparer(
        source_dir=source_dir,
        output_dir=output_dir,
        target_hours=hours
    )
    
    dataset_info = preparer.prepare_dataset()
    print(f"\nDataset prepared: {len(dataset_info)} videos")
    
    return dataset_info

def run_zbuffer_training(data_dir, dataset_info_path, output_dir, epochs):
    print("\n" + "=" * 60)
    print("STEP 2: Training Z-buffer Model")
    print("=" * 60)
    
    from train_zbuffer import train_zbuffer_model
    
    results = train_zbuffer_model(
        data_dir=data_dir,
        dataset_info_path=dataset_info_path,
        output_dir=output_dir,
        num_epochs=epochs,
        batch_size=1
    )
    
    print(f"\nZ-buffer training complete: {len(results)} epochs")
    
    return results

def run_peekaboo_training(data_dir, dataset_info_path, output_dir, epochs):
    print("\n" + "=" * 60)
    print("STEP 3: Training Peekaboo Baseline")
    print("=" * 60)
    
    from train_peekaboo import train_peekaboo_model
    
    results = train_peekaboo_model(
        data_dir=data_dir,
        dataset_info_path=dataset_info_path,
        output_dir=output_dir,
        num_epochs=epochs,
        batch_size=1
    )
    
    print(f"\nPeekaboo training complete: {len(results)} epochs")
    
    return results

def run_evaluation(zbuffer_dir, peekaboo_dir, dataset_info_path, output_dir):
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating and Comparing")
    print("=" * 60)
    
    from evaluate_comparison import evaluate_and_compare
    
    comparison = evaluate_and_compare(
        zbuffer_dir=zbuffer_dir,
        peekaboo_dir=peekaboo_dir,
        dataset_info_path=dataset_info_path,
        output_dir=output_dir
    )
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")
    
    return comparison

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, required=True, help='Animal Kingdom source directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--hours', type=float, default=3.5, help='Hours of data to use')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--skip_data_prep', action='store_true', help='Skip data preparation')
    parser.add_argument('--skip_training', action='store_true', help='Skip training')
    parser.add_argument('--skip_evaluation', action='store_true', help='Skip evaluation')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = output_dir / 'animal_kingdom_subset'
    dataset_info_path = data_dir / 'dataset_info.json'
    zbuffer_dir = output_dir / 'zbuffer_training'
    peekaboo_dir = output_dir / 'peekaboo_training'
    eval_dir = output_dir / 'evaluation'
    
    if not args.skip_data_prep:
        run_data_preparation(args.source_dir, data_dir, args.hours)
    
    if not args.skip_training:
        run_zbuffer_training(data_dir, dataset_info_path, zbuffer_dir, args.epochs)
        run_peekaboo_training(data_dir, dataset_info_path, peekaboo_dir, args.epochs)
    
    if not args.skip_evaluation:
        run_evaluation(zbuffer_dir, peekaboo_dir, dataset_info_path, eval_dir)
    
    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved in: {output_dir}")
    print(f"  - Dataset: {data_dir}")
    print(f"  - Z-buffer outputs: {zbuffer_dir}")
    print(f"  - Peekaboo outputs: {peekaboo_dir}")
    print(f"  - Evaluation: {eval_dir}")

if __name__ == "__main__":
    main()