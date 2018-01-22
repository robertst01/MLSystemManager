// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

using System;
using System.IO;

namespace MLSystemManager.Algorithms
{
	public abstract class SupervisedLearner
	{
		// true if verbose
		public Parameters Parameters { get; set; }

		// Before you call this method, you need to divide your data
		// into a feature matrix and a label matrix.
		public abstract void Train(Matrix features, Matrix labels, double[] colMin, double[] colMax);
		public abstract void VTrain(VMatrix features, VMatrix labels, double[] colMin, double[] colMax);

		// A feature vector goes in. A label vector comes out. (Some supervised
		// learning algorithms only support one-dimensional label vectors. Some
		// support multi-dimensional label vectors.)
		public abstract void Predict(double[] features, double[] labels);

		// The model must be trained before you call this method.
		public virtual double GetMSE(Matrix features, Matrix labels)
		{
			return 0;
		}

		public virtual double VGetMSE(VMatrix features, VMatrix labels)
		{
			return 0;
		}

		// The model must be trained before you call this method. If the label is nominal,
		// it returns the predictive accuracy. If the label is continuous, it returns
		// the root mean squared error (RMSE). If confusion is non-NULL, and the
		// output label is nominal, then confusion will hold stats for a confusion matrix.
		public double MeasureAccuracy(Matrix features, Matrix labels, Matrix confusion)
		{
			if (features.Rows() != labels.Rows())
			{
				throw (new Exception("Expected the features and labels to have the same number of rows"));
			}
			if (labels.Cols() != 1)
			{
				throw (new Exception("Sorry, this method currently only supports one-dimensional labels"));
			}
			if (features.Rows() == 0)
			{
				throw (new Exception("Expected at least one row"));
			}

			var labelValues = labels.ValueCount(0);
			if (labelValues == 0) // If the label is continuous...
			{
				// The label is continuous, so measure root mean squared error
				var pred = new double[1];
				var sse = 0.0;
				for (var i = 0; i < features.Rows(); i++)
				{
					var feat = features.Row(i);
					var targ = labels.Row(i);
					pred[0] = 0.0; // make sure the prediction is not biassed by a previous prediction
					Predict(feat, pred);
					var delta = targ[0] - pred[0];
					sse += (delta * delta);
				}
				return Math.Sqrt(sse / features.Rows());
			}
			else
			{
				// The label is nominal, so measure predictive accuracy
				if (confusion != null)
				{
					confusion.SetSize(labelValues, labelValues);
					for (var i = 0; i < labelValues; i++)
					{
						confusion.SetAttrName(i, labels.AttrValue(0, i));
					}
				}
				var correctCount = 0;
				var prediction = new double[1];
				for (var i = 0; i < features.Rows(); i++)
				{
					var feat = features.Row(i);
					var targ = (int)labels.Get(i, 0);
					if (targ >= labelValues)
					{
						throw new Exception("The label is out of range");
					}
					Predict(feat, prediction);
					var pred = (int)prediction[0];
					if (confusion != null)
					{
						confusion.Set(targ, pred, confusion.Get(targ, pred) + 1);
					}
					if (pred == targ)
					{
						correctCount++;
					}
				}
				return (double)correctCount / features.Rows();
			}
		}
		
		public double VMeasureAccuracy(VMatrix features, VMatrix labels, Matrix confusion)
		{
			if (features.Rows() != labels.Rows())
			{
				throw (new Exception("Expected the features and labels to have the same number of rows"));
			}
			if (labels.Cols() != 1)
			{
				throw (new Exception("Sorry, this method currently only supports one-dimensional labels"));
			}
			if (features.Rows() == 0)
			{
				throw (new Exception("Expected at least one row"));
			}

			var cl = 0;
			if (Parameters.Verbose)
			{
				Console.Write("VMeasureAccuracy ");
				cl = Console.CursorLeft;
			}

			var count = features.Rows();
			var begRow = 0;
			if (this is BPTT)
			{
				var learner = this as BPTT;
				begRow = learner.m_k - 1;
				count -= begRow;
			}

			var labelValues = labels.ValueCount(0);
			if (labelValues == 0) // If the label is continuous...
			{
				// The label is continuous, so measure root mean squared error
				var pred = new double[1];
				var sse = 0.0;
				for (var i = 0; i < features.Rows(); i++)
				{
					if (Parameters.Verbose)
					{
						Console.SetCursorPosition(cl, Console.CursorTop);
						Console.Write(i);
					}

					var feat = features.Row(i);
					var targ = labels.Row(i);
					pred[0] = 0.0; // make sure the prediction is not biased by a previous prediction
					Predict(feat, pred);
					if (i >= begRow)
					{
						var delta = targ[0] - pred[0];
						sse += (delta * delta);
					}
				}

				if (Parameters.Verbose)
				{
					Console.WriteLine();
				}

				return Math.Sqrt(sse / count);
			}
			else
			{
				// The label is nominal, so measure predictive accuracy
				if (confusion != null)
				{
					confusion.SetSize(labelValues, labelValues);
					for (var i = 0; i < labelValues; i++)
					{
						confusion.SetAttrName(i, labels.AttrValue(0, i));
					}
				}
				var correctCount = 0;
				var prediction = new double[1];
				for (var i = 0; i < features.Rows(); i++)
				{
					if (Parameters.Verbose)
					{
						Console.SetCursorPosition(cl, Console.CursorTop);
						Console.Write(i);
					}

					var feat = features.Row(i);
					var lab = labels.Get(i, 0);
					if (lab != Matrix.MISSING)
					{
						var targ = (int)lab;
						if (targ >= labelValues)
						{
							throw new Exception("The label is out of range");
						}
						Predict(feat, prediction);
						if (i >= begRow)
						{
							var pred = (int)prediction[0];
							if (confusion != null)
							{
								confusion.Set(targ, pred, confusion.Get(targ, pred) + 1);
							}
							if (pred == targ)
							{
								correctCount++;
							}
						}
					}
					else
					{
						count--;
					}
				}

				if (Parameters.Verbose)
				{
					Console.WriteLine();
				}

				return (double)correctCount / count;
			}
		}
	}
}