"""
Model evaluation orchestrator
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from .metrics import IrisSegmentationMetrics, benchmark_inference_speed


class ModelEvaluator:
    """
    Comprehensive model evaluation for iris segmentation
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int = 2,
        save_predictions: bool = False,
        output_dir: Optional[str] = None
    ):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.save_predictions = save_predictions
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'predictions').mkdir(exist_ok=True)
            (self.output_dir / 'failed_cases').mkdir(exist_ok=True)
        
        self.metrics = IrisSegmentationMetrics(num_classes)
    
    def evaluate(
        self,
        dataloader: DataLoader,
        save_failed_cases: bool = True,
        iou_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Evaluate model on given dataloader
        
        Args:
            dataloader: DataLoader for evaluation
            save_failed_cases: Whether to save poorly performing samples
            iou_threshold: IoU threshold for determining failed cases
        
        Returns:
            Dictionary with comprehensive evaluation results
        """
        self.model.eval()
        self.metrics.reset()
        
        failed_cases = []
        all_predictions = []
        
        print(f"Evaluating model on {len(dataloader)} batches...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                # Move to device
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                boundary = batch.get('boundary', None)
                if boundary is not None:
                    boundary = boundary.to(self.device)
                
                # Forward pass
                outputs = self.model(pixel_values, return_boundary=True)
                predictions = torch.argmax(outputs['logits'], dim=1)
                
                # Update metrics
                boundary_preds = outputs.get('boundary_logits', None)
                self.metrics.update(predictions, labels, boundary_preds, boundary)
                
                # Check for failed cases
                if save_failed_cases:
                    batch_ious = self._compute_batch_ious(predictions, labels)
                    
                    for i, iou in enumerate(batch_ious):
                        if iou < iou_threshold:
                            failed_cases.append({
                                'batch_idx': batch_idx,
                                'sample_idx': i,
                                'iou': iou,
                                'image_path': batch.get('image_path', [None])[i],
                                'prediction': predictions[i].cpu(),
                                'target': labels[i].cpu(),
                                'image': pixel_values[i].cpu()
                            })
                
                # Save predictions if requested
                if self.save_predictions:
                    for i in range(predictions.shape[0]):
                        pred_data = {
                            'prediction': predictions[i].cpu(),
                            'target': labels[i].cpu(),
                            'image': pixel_values[i].cpu(),
                            'batch_idx': batch_idx,
                            'sample_idx': i
                        }
                        all_predictions.append(pred_data)
        
        # Compute final metrics
        final_metrics = self.metrics.compute_all_metrics()
        
        # Add evaluation summary
        evaluation_results = {
            'metrics': final_metrics,
            'total_samples': len(dataloader.dataset),
            'failed_cases': len(failed_cases),
            'failed_rate': len(failed_cases) / len(dataloader.dataset),
            'evaluation_summary': self._create_evaluation_summary(final_metrics)
        }
        
        # Save failed cases
        if save_failed_cases and failed_cases and self.output_dir:
            self._save_failed_cases(failed_cases, iou_threshold)
        
        # Save predictions
        if self.save_predictions and all_predictions and self.output_dir:
            self._save_predictions(all_predictions)
        
        return evaluation_results
    
    def _compute_batch_ious(self, predictions: torch.Tensor, targets: torch.Tensor) -> List[float]:
        """Compute IoU for each sample in batch"""
        batch_size = predictions.shape[0]
        ious = []
        
        for i in range(batch_size):
            pred = predictions[i].cpu().numpy()
            target = targets[i].cpu().numpy()
            
            # Compute IoU for iris class (class 1)
            pred_iris = (pred == 1)
            target_iris = (target == 1)
            
            intersection = (pred_iris & target_iris).sum()
            union = (pred_iris | target_iris).sum()
            
            iou = intersection / union if union > 0 else 0.0
            ious.append(float(iou))
        
        return ious
    
    def _save_failed_cases(self, failed_cases: List[Dict], iou_threshold: float):
        """Save visualizations of failed cases"""
        print(f"Saving {len(failed_cases)} failed cases (IoU < {iou_threshold})...")
        
        for i, case in enumerate(failed_cases[:20]):  # Save top 20 worst cases
            self._visualize_case(
                case,
                save_path=self.output_dir / 'failed_cases' / f'failed_case_{i+1}_iou_{case["iou"]:.3f}.png'
            )
    
    def _save_predictions(self, predictions: List[Dict]):
        """Save all predictions"""
        print(f"Saving {len(predictions)} predictions...")
        
        # Save every 10th prediction to avoid too many files
        for i, pred_data in enumerate(predictions[::10]):
            self._visualize_case(
                pred_data,
                save_path=self.output_dir / 'predictions' / f'prediction_{i+1}.png'
            )
    
    def _visualize_case(self, case_data: Dict, save_path: Path):
        """Visualize a single prediction case"""
        image = case_data['image']
        prediction = case_data['prediction']
        target = case_data['target']
        
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        image = image.permute(1, 2, 0).numpy()
        
        prediction = prediction.numpy()
        target = target.numpy()
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(target, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(prediction, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Error map
        error_map = np.zeros((*prediction.shape, 3))
        error_map[prediction != target] = [1, 0, 0]  # Red for errors
        error_map[prediction & target] = [0, 1, 0]   # Green for correct positives
        
        axes[3].imshow(error_map)
        axes[3].set_title('Error Map\n(Red=Error, Green=Correct)')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_evaluation_summary(self, metrics: Dict[str, float]) -> str:
        """Create human-readable evaluation summary"""
        summary = f"""
Evaluation Summary:
==================
Overall Performance:
  - Pixel Accuracy: {metrics.get('pixel_accuracy', 0):.3f}
  - Mean IoU: {metrics.get('mean_iou', 0):.3f}
  - Mean Dice: {metrics.get('mean_dice', 0):.3f}

Class-wise Performance:
  Background/Pupil (Class 0):
    - IoU: {metrics.get('class_0_iou', 0):.3f}
    - Dice: {metrics.get('class_0_dice', 0):.3f}
    - Precision: {metrics.get('class_0_precision', 0):.3f}
    - Recall: {metrics.get('class_0_recall', 0):.3f}
    - F1: {metrics.get('class_0_f1', 0):.3f}
  
  Iris (Class 1):
    - IoU: {metrics.get('class_1_iou', 0):.3f}
    - Dice: {metrics.get('class_1_dice', 0):.3f}
    - Precision: {metrics.get('class_1_precision', 0):.3f}
    - Recall: {metrics.get('class_1_recall', 0):.3f}
    - F1: {metrics.get('class_1_f1', 0):.3f}

Boundary Performance:
  - Boundary F1: {metrics.get('boundary_f1', 0):.3f}

Performance Assessment:
  - {'✅ Excellent' if metrics.get('mean_iou', 0) > 0.9 else '🟡 Good' if metrics.get('mean_iou', 0) > 0.8 else '❌ Needs Improvement'}
  - Target: mIoU ≥ 0.90, Dice ≥ 0.93
"""
        return summary
    
    def benchmark_speed(
        self,
        input_size: tuple = (1, 3, 512, 512),
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark model inference speed"""
        return benchmark_inference_speed(
            self.model,
            input_size=input_size,
            device=self.device,
            num_runs=num_runs
        )


class CrossValidationEvaluator:
    """
    Cross-validation evaluation for robust performance estimation
    """
    
    def __init__(
        self,
        model_factory: callable,
        n_folds: int = 5,
        device: torch.device = torch.device('cpu'),
        output_dir: Optional[str] = None
    ):
        self.model_factory = model_factory
        self.n_folds = n_folds
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_fold(
        self,
        fold_idx: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        checkpoint_path: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single fold
        
        Args:
            fold_idx: Fold index
            train_loader: Training data loader
            val_loader: Validation data loader  
            test_loader: Test data loader
            checkpoint_path: Path to trained model checkpoint
        
        Returns:
            Fold evaluation results
        """
        print(f"Evaluating fold {fold_idx + 1}/{self.n_folds}...")
        
        # Load trained model
        model = self.model_factory()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        # Create evaluator
        fold_output_dir = self.output_dir / f'fold_{fold_idx}' if self.output_dir else None
        evaluator = ModelEvaluator(
            model=model,
            device=self.device,
            save_predictions=True,
            output_dir=fold_output_dir
        )
        
        # Evaluate on test set
        test_results = evaluator.evaluate(test_loader, save_failed_cases=True)
        
        # Benchmark speed
        speed_results = evaluator.benchmark_speed()
        
        return {
            'fold_idx': fold_idx,
            'test_metrics': test_results['metrics'],
            'speed_metrics': speed_results,
            'evaluation_summary': test_results['evaluation_summary'],
            'checkpoint_path': checkpoint_path
        }
    
    def evaluate_all_folds(
        self,
        fold_results: List[str],  # List of checkpoint paths
        test_loaders: List[DataLoader]
    ) -> Dict[str, Any]:
        """
        Evaluate all cross-validation folds
        
        Args:
            fold_results: List of checkpoint paths for each fold
            test_loaders: List of test data loaders for each fold
        
        Returns:
            Aggregated cross-validation results
        """
        assert len(fold_results) == len(test_loaders) == self.n_folds
        
        fold_metrics = []
        
        for fold_idx, (checkpoint_path, test_loader) in enumerate(zip(fold_results, test_loaders)):
            fold_result = self.evaluate_fold(
                fold_idx, None, None, test_loader, checkpoint_path
            )
            fold_metrics.append(fold_result)
        
        # Aggregate results
        aggregated_results = self._aggregate_fold_results(fold_metrics)
        
        # Save aggregated results
        if self.output_dir:
            self._save_cv_results(aggregated_results)
        
        return aggregated_results
    
    def _aggregate_fold_results(self, fold_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across all folds"""
        
        # Collect metrics from all folds
        all_metrics = {}
        metric_names = fold_metrics[0]['test_metrics'].keys()
        
        for metric_name in metric_names:
            values = [fold['test_metrics'][metric_name] for fold in fold_metrics]
            all_metrics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        # Aggregate speed metrics
        speed_metrics = {}
        speed_names = fold_metrics[0]['speed_metrics'].keys()
        
        for speed_name in speed_names:
            values = [fold['speed_metrics'][speed_name] for fold in fold_metrics]
            speed_metrics[speed_name] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        return {
            'n_folds': self.n_folds,
            'aggregated_metrics': all_metrics,
            'aggregated_speed': speed_metrics,
            'fold_results': fold_metrics,
            'summary': self._create_cv_summary(all_metrics)
        }
    
    def _create_cv_summary(self, aggregated_metrics: Dict) -> str:
        """Create cross-validation summary"""
        miou_mean = aggregated_metrics.get('mean_iou', {}).get('mean', 0)
        miou_std = aggregated_metrics.get('mean_iou', {}).get('std', 0)
        
        dice_mean = aggregated_metrics.get('mean_dice', {}).get('mean', 0)
        dice_std = aggregated_metrics.get('mean_dice', {}).get('std', 0)
        
        iris_iou_mean = aggregated_metrics.get('class_1_iou', {}).get('mean', 0)
        iris_iou_std = aggregated_metrics.get('class_1_iou', {}).get('std', 0)
        
        summary = f"""
Cross-Validation Results ({self.n_folds}-Fold):
=============================================

Key Metrics (Mean ± Std):
  - Mean IoU: {miou_mean:.3f} ± {miou_std:.3f}
  - Mean Dice: {dice_mean:.3f} ± {dice_std:.3f}
  - Iris IoU: {iris_iou_mean:.3f} ± {iris_iou_std:.3f}

Performance Assessment:
  - {'✅ Excellent and Robust' if miou_mean > 0.9 and miou_std < 0.02 else '🟡 Good Performance' if miou_mean > 0.8 else '❌ Needs Improvement'}
  - Target: mIoU ≥ 0.90 ± 0.02

Variance Analysis:
  - {'Low variance (good generalization)' if miou_std < 0.02 else 'High variance (potential overfitting)' if miou_std > 0.05 else 'Moderate variance'}
"""
        return summary
    
    def _save_cv_results(self, results: Dict[str, Any]):
        """Save cross-validation results"""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if key == 'aggregated_metrics':
                serializable_results[key] = {}
                for metric_name, metric_data in value.items():
                    serializable_results[key][metric_name] = {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else 
                           [float(x) for x in v] if isinstance(v, (list, np.ndarray)) else v
                        for k, v in metric_data.items()
                    }
            elif isinstance(value, (np.floating, np.integer)):
                serializable_results[key] = float(value)
            elif key == 'fold_results':
                # Skip detailed fold results to keep file size manageable
                continue
            else:
                serializable_results[key] = value
        
        # Save results
        results_path = self.output_dir / 'cv_results.json'
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save summary
        summary_path = self.output_dir / 'cv_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(results['summary'])
        
        print(f"Cross-validation results saved to {self.output_dir}")
