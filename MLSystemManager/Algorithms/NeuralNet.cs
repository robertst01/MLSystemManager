#define parallel

using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace MLSystemManager.Algorithms
{
	public class NeuralNet : SupervisedLearner
	{
		private readonly List<Layer> _layers;
		private readonly Random _rand;
		private readonly double _rate = 0.1;
		private readonly double _momentum = 0.9;
		private readonly string _activation = "sigmoid";
		private int[] _hidden;
		private StreamWriter _outputFile;

		public enum LayerType
		{
			Input,
			Hidden,
			Output
		}

		public class Layer
		{
			public LayerType Type { get; set; }
			public List<Node> Nodes { get; set; }
			public Layer Previous { get; set; }
			public Layer Next { get; set; }
		}

		public class Node
		{
			public int Index { get; set; }					// the index of this node
			public double Net { get; set; }					// the net for this node
			public double Output { get; set; }				// the output of this node
			public double Error { get; set; }				// the error for this node
			public double[] Weights { get; set; }			// the weights for all nodes connected to this node
			public double[] BestWeights { get; set; }		// the best weights so far
			public double[] Deltas { get; set; }			// weight deltas from previous epoch

			public Node()
			{
				Index = 0;
				Net = 0;
				Output = 0;
				Error = 0;
				Weights = null;
				BestWeights = null;
				Deltas = null;
			}

			public Node(int idx, int numWeights, Random rand)
			{
				Index = idx;

				if (numWeights > 0)
				{
					Weights = new double[numWeights];
					BestWeights = new double[numWeights];
					for (var i = 0; i < numWeights; i++)
					{
						Weights[i] = 0.1 - (rand.NextDouble() * 0.2);
						BestWeights[i] = Weights[i];
					}

					Deltas = new double[numWeights];
					for (var i = 0; i < numWeights; i++)
					{
						Deltas[i] = 0;
					}
				}
			}

			public void SaveBestWeights()
			{
				if (Weights != null)
				{
					for (var i = 0; i < Weights.Length; i++)
					{
						BestWeights[i] = Weights[i];
					}
				}
			}

			public void RestoreBestWeights()
			{
				if (Weights != null)
				{
					for (var i = 0; i < Weights.Length; i++)
					{
						Weights[i] = BestWeights[i];
					}
				}
			}
		}

		public class InputNode : Node
		{
			public int Feature { get; set; }
			public double MinValue { get; set; }
			public double MaxValue { get; set; }
			public InputNode(int idx, int feature, double minValue, double maxValue, Random rand)
				: base(idx, 0, rand)
			{
				Feature = feature;
				MinValue = minValue;
				MaxValue = maxValue;
			}
		}

		public class HiddenNode : Node
		{
			public HiddenNode(int idx, int numWeights, Random rand)
				: base(idx, numWeights, rand)
			{
			}
		}

		public class OutputNode : Node
		{
			public bool IsContinuous { get; set; }			// true if the outout is continuous
			public int LabelCol { get; set; }				// the label column that this output node corresponds to
			public double LabelVal { get; set; }			// the value of the label column that this output node corresponds to
			public OutputNode(int idx, int numWeights, bool isContinuous, int labelCol, double labelVal, Random rand)
				: base(idx, numWeights, rand)
			{
				IsContinuous = isContinuous;
				LabelCol = labelCol;
				LabelVal = labelVal;
			}
		}

		public NeuralNet()
		{
			_rand = new Random();
			_layers = new List<Layer>();
		}

		public NeuralNet(Parameters parameters)
		{
			_rand = Rand.Get();
			_rate = parameters.Rate;
			_momentum = parameters.Momentum;
			_activation = parameters.Activation;
			_hidden = parameters.Hidden;
			_layers = new List<Layer>();
		}

		public override void Train(Matrix features, Matrix labels, double[] colMin, double[] colMax)
		{
		}

		public override void VTrain(VMatrix features, VMatrix labels, double[] colMin, double[] colMax)
		{
			if (_hidden.Length < 1)
			{
				_hidden = new[] { features.Cols() * 2 };
			}

			// add the input nodes
			var iNodes = new List<Node>();
			for (var i = 0; i < features.Cols(); i++)
			{
				iNodes.Add(new InputNode(i, i, colMin[i], colMax[i], _rand));
			}

			var iLayer = new Layer()
			{
				Type = LayerType.Input,
				Nodes = iNodes,
				Previous = null,
				Next = null
			};
			_layers.Add(iLayer);

			var prevLayer = iLayer;

			// add the hidden nodes
			foreach (var t in _hidden)
			{
				var hNodes = new List<Node>();

				for (var n = 0; n < t; n++)
				{
					var node = new HiddenNode(n, prevLayer.Nodes.Count + 1, _rand);
					hNodes.Add(node);
				}

				var hLayer = new Layer()
				{
					Type = LayerType.Hidden,
					Nodes = hNodes,
					Previous = prevLayer,
					Next = null
				};
				_layers.Add(hLayer);

				prevLayer.Next = hLayer;
				prevLayer = hLayer;
			}

			// add the output nodes
			var oNodes = new List<Node>();
			for (var col = 0; col < labels.Cols(); col++)
			{
				var labelValueCount = labels.ValueCount(col);

				if (labelValueCount < 2)
				{
					// continuous
					var node = new OutputNode(oNodes.Count, prevLayer.Nodes.Count + 1, true, col, -1, _rand);
					oNodes.Add(node);
				}
				else
				{
					for (var n = 0; n < labelValueCount; n++)
					{
						var node = new OutputNode(oNodes.Count, prevLayer.Nodes.Count + 1, false, col, n, _rand);
						oNodes.Add(node);
					}
				}
			}

			var oLayer = new Layer()
			{
				Type = LayerType.Output,
				Nodes = oNodes,
				Previous = prevLayer
			};
			_layers.Add(oLayer);

			prevLayer.Next = oLayer;

			if (!string.IsNullOrEmpty(OutputFileName))
			{
				_outputFile = File.AppendText(OutputFileName);
			}

			var trainSize = (int)(0.75 * features.Rows());
			var trainFeatures = new VMatrix(features, 0, 0, trainSize, features.Cols());
			var trainLabels = new VMatrix(labels, 0, 0, trainSize, labels.Cols());
			var validationFeatures = new VMatrix(features, trainSize, 0, features.Rows() - trainSize, features.Cols());
			var validationLabels = new VMatrix(labels, trainSize, 0, labels.Rows() - trainSize, labels.Cols());

			Console.Write("Layers: ");
			Console.Write(iNodes.Count);
			Console.Write('x');
			foreach (var t in _hidden)
			{
				Console.Write(t);
				Console.Write('x');
			}
			Console.WriteLine(oNodes.Count);
			
			Console.WriteLine("AF: " + _activation);
			
			Console.WriteLine("Epoch\tMSE (validation)");
			if (_outputFile != null)
			{
				_outputFile.Write("Layers: ");
				_outputFile.Write(iNodes.Count);
				_outputFile.Write('x');
				foreach (var t in _hidden)
				{
					_outputFile.Write(t);
					_outputFile.Write('x');
				}
				_outputFile.WriteLine(oNodes.Count);

				_outputFile.WriteLine("Momentum: " + _momentum);
				_outputFile.WriteLine("AF: " + _activation);
				_outputFile.WriteLine("Weights");
				PrintWeights();
				_outputFile.WriteLine("Epoch\tMSE (validation)");
			}

			int epoch;								// current epoch number
			var bestEpoch = 0;						// epoch number of best MSE
			var eCount = 0;							// number of epochs since the best MSE
			var checkDone = false;					// if true, check to see if we're done
			var initialMse = double.MaxValue;		// MSE for first epoch
			var bestMse = double.MaxValue;			// best validation MSE so far

			for (epoch = 1; ; epoch++)
			{
				// shuffle the training set
				trainFeatures.Shuffle(_rand, trainLabels);

				TrainEpoch(trainFeatures, trainLabels);

				// check the MSE after this epoch
				var mse = VGetMSE(validationFeatures, validationLabels);

				Console.WriteLine($"{epoch}-{eCount}\t{mse}");
				if (_outputFile != null)
				{
					_outputFile.WriteLine($"{epoch}-{eCount}\t{mse}");
					_outputFile.Flush();
				}

				if ((mse == 0) || (epoch > 5000))
				{
					break;
				}

				if ((epoch == 1) || (mse < bestMse))
				{
					if (epoch == 1)
					{
						// save the initial MSE
						initialMse = mse;
					}
					else if (!checkDone && (mse < initialMse * 0.9))
					{
						checkDone = true;
					}
					eCount = 0;

					// save the best for later
					bestMse = mse;
					bestEpoch = epoch;
					foreach (var layer in _layers)
					{
						foreach (var node in layer.Nodes)
						{
							node.SaveBestWeights();
						}
					}
				}
				else if (checkDone)
				{
					// check to see if we're done
					eCount++;
					if (eCount >= 20)
					{
						break;
					}
				}
			}

			if (_outputFile != null)
			{
				_outputFile.WriteLine();
				_outputFile.WriteLine("Weights");
				PrintWeights();
			}

			if ((bestEpoch > 0) && (bestEpoch != epoch))
			{
				foreach (var layer in _layers)
				{
					foreach (var node in layer.Nodes)
					{
						node.RestoreBestWeights();
					}
				}
				if (_outputFile != null)
				{
					_outputFile.WriteLine();
					_outputFile.WriteLine($"Best Weights (from Epoch {bestEpoch}, valMSE={bestMse})");
					PrintWeights();
				}
			}

			if (_outputFile != null)
			{
				_outputFile.Close();
			}
		}

		private void TrainEpoch(VMatrix features, VMatrix labels)
		{
			Console.Write("TrainEpoch ");
			var cl = Console.CursorLeft;

			for (var row = 0; row < features.Rows(); row++)
			{
				if (((row % 100) == 0) || (row == (features.Rows() - 1)))
				{
					Console.SetCursorPosition(cl, Console.CursorTop);
					Console.Write(row);
				}

				// calculate the output
				foreach (var layer in _layers)
				{
#if parallel
					Parallel.ForEach(layer.Nodes, node =>
#else
					foreach (var node in layer.Nodes)
#endif
					{
						node.Net = 0;
						node.Output = 0;
						node.Error = 0;

						if (layer.Type == LayerType.Input)
						{
							// input node
							node.Net = features.Get(row, node.Index);
							node.Output = node.Net;
						}
						else
						{
							// calculate the net value
							for (var w = 0; w < node.Weights.Length - 1; w++)
							{
								node.Net += node.Weights[w] * layer.Previous.Nodes[w].Output;
							}
							// add the bias
							node.Net += node.Weights[node.Weights.Length - 1];

							// calculate the output
							switch (_activation)
							{
								case "relu":
									node.Output = node.Net < 0 ? 0.01 * node.Net : node.Net;
									break;
								case "softsign":
									node.Output = (node.Net / (1.0 + Math.Abs(node.Net)));
									break;
								case "softplus":
									node.Output = Math.Log(1.0 + Math.Exp(node.Net));
									break;
								default:
									node.Output = 1.0 / (1.0 + Math.Exp(-node.Net));
									break;
							}
						}
#if parallel
					});
#else
					}
#endif
				}

				// calculate the error and weight changes
				for (var i = _layers.Count - 1; i > 0; i--)
				{
					var layer = _layers[i];
#if parallel
					Parallel.ForEach(layer.Nodes, node =>
#else
					foreach (var node in layer.Nodes)
#endif
					{
						double fPrime;
						switch (_activation)
						{
							case "relu":
								fPrime = (node.Output < 0 ? 0.01 : 1);
								break;
							case "softsign":
								fPrime = 1.0 / ((1.0 - Math.Abs(node.Output)) * (1.0 - Math.Abs(node.Output)));
								break;
							case "softplus":
								fPrime = 1.0 / (1.0 + Math.Exp(-node.Output));
								break;
							default:
								fPrime = node.Output * (1.0 - node.Output);
								break;
						}

						if (layer.Type == LayerType.Output)
						{
							// output layer
							var oNode = node as OutputNode;
							var target = labels.Get(row, oNode.LabelCol);
							if (!oNode.IsContinuous)
							{
								// nominal
								target = target == oNode.LabelVal ? 0.99 : 0.01;
							}

							var error = target - node.Output;
							node.Error = error * fPrime;
						}
						else
						{
							// hidden layer
							double sum = 0;
							foreach (var tn in layer.Next.Nodes)
							{
								sum += tn.Error * tn.Weights[node.Index];
							}
							node.Error = sum * fPrime;
						}

						// calculate the weight changes
						double delta;
						for (var w = 0; w < node.Weights.Length - 1; w++)
						{
							delta = _rate * node.Error * layer.Previous.Nodes[w].Output;
							delta += _momentum * node.Deltas[w];
							node.Deltas[w] = delta;
						}

						// calculate the bias weight change
						delta = _rate * node.Error;
						delta += _momentum * node.Deltas[node.Weights.Length - 1];
						node.Deltas[node.Weights.Length - 1] = delta;
#if parallel
					});
#else
					}
#endif
				}

				// update the weights
				foreach (var layer in _layers)
				{
					if (layer.Type != LayerType.Input)
					{
#if parallel
						Parallel.ForEach(layer.Nodes, node =>
#else
						foreach (var node in layer.Nodes)
#endif
						{
							for (var w = 0; w < node.Weights.Length; w++)
							{
								node.Weights[w] += node.Deltas[w];
							}
#if parallel
						});
#else
						}
#endif
					}
				}
			}

			Console.WriteLine();
		}

		// Calculate the MSE
		public override double GetMSE(Matrix features, Matrix labels)
		{
			return 0;
		}

		// Calculate the MSE
		public override double VGetMSE(VMatrix features, VMatrix labels)
		{
			double sse = 0;

			Console.Write("VGetMSE ");
			var cl = Console.CursorLeft;

			for (var row = 0; row < features.Rows(); row++)
			{
				if (((row % 10) == 0) || (row == (features.Rows() - 1)))
				{
					Console.SetCursorPosition(cl, Console.CursorTop);
					Console.Write(row);
				}

				// calculate the output
				foreach (var layer in _layers)
				{
#if parallel
					Parallel.ForEach(layer.Nodes, node =>
#else
					foreach (var node in layer.Nodes)
#endif
					{
						node.Net = 0;
						node.Output = 0;

						if (layer.Type == LayerType.Input)
						{
							// input node
							node.Output = features.Get(row, node.Index);
						}
						else
						{
							// calculate the net value
							for (var w = 0; w < node.Weights.Length - 1; w++)
							{
								node.Net += node.Weights[w] * layer.Previous.Nodes[w].Output;
							}
							// add the bias
							node.Net += node.Weights[node.Weights.Length - 1];

							// calculate the output
							switch (_activation)
							{
								case "relu":
									node.Output = node.Net < 0 ? 0.01 * node.Net : node.Net;
									break;
								case "softsign":
									node.Output = (node.Net / (1.0 + Math.Abs(node.Net)));
									break;
								case "softplus":
									node.Output = Math.Log(1.0 + Math.Exp(node.Net));
									break;
								default:
									node.Output = 1.0 / (1.0 + Math.Exp(-node.Net));
									break;
							}
						}
#if parallel
					});
#else
					}
#endif
				}

				// calculate the error of the output layer
				foreach (var node in _layers[_layers.Count - 1].Nodes)
				{
					var oNode = node as OutputNode;
					var target = labels.Get(row, oNode.LabelCol);
					if (!oNode.IsContinuous)
					{
						// nominal
						if (target == oNode.LabelVal)
						{
							target = 0.99;
						}
						else
						{
							target = 0.01;
						}
					}
					var error = target - node.Output;

					// update the error
					sse += error * error;
				}
			}

			Console.WriteLine();

			return sse / features.Rows();
		}

		private void PrintWeights()
		{
			foreach (var layer in _layers)
			{
				if (layer.Type == LayerType.Input)
				{
					continue;
				}
				_outputFile.WriteLine("Layer " + _layers.IndexOf(layer));
				foreach (var node in layer.Nodes)
				{
					for (var w = 0; w < node.Weights.Length - 1; w++)
					{
						_outputFile.Write($"{node.Weights[w]}\t");
					}
					_outputFile.WriteLine(node.Weights[node.Weights.Length - 1]);
				}
			}
			_outputFile.WriteLine();
			_outputFile.Flush();
		}

		public override void Predict(double[] features, double[] labels)
		{
			foreach (var layer in _layers)
			{
#if parallel
				Parallel.ForEach(layer.Nodes, node =>
#else
				foreach (var node in layer.Nodes)
#endif
				{
					node.Net = 0;
					node.Output = 0;

					if (layer.Type == LayerType.Input)
					{
						// input node
						node.Output = features[node.Index];
					}
					else
					{
						// calculate the net value
						for (var w = 0; w < node.Weights.Length - 1; w++)
						{
							node.Net += node.Weights[w] * layer.Previous.Nodes[w].Output;
						}
						// add the bias
						node.Net += node.Weights[node.Weights.Length - 1];

						switch (_activation)
						{
							case "relu":
								node.Output = node.Net < 0 ? 0.01 * node.Net : node.Net;
								break;
							case "softsign":
								node.Output = (node.Net / (1.0 + Math.Abs(node.Net)));
								break;
							case "softplus":
								node.Output = Math.Log(1.0 + Math.Exp(node.Net));
								break;
							default:
								node.Output = 1.0 / (1.0 + Math.Exp(-node.Net));
								break;
						}
					}
#if parallel
				});
#else
				}
#endif
			}

			var labelIdx = 0;
			var outputLayer = _layers[_layers.Count - 1];
			for (var n = 0; n < outputLayer.Nodes.Count; n++)
			{
				var oNode = outputLayer.Nodes[n] as OutputNode;

				if (oNode.IsContinuous)
				{
					labels[labelIdx++] = oNode.Output;
				}
				else
				{
					// find the max output for this labelCol
					var max = oNode.Output;
					var labelCol = oNode.LabelCol;
					var labelVal = oNode.LabelVal;
					int nIdx;
					for (nIdx = 1; nIdx + n < outputLayer.Nodes.Count; nIdx++)
					{
						var tn = outputLayer.Nodes[n + nIdx] as OutputNode;
						if (tn.LabelCol != labelCol)
						{
							break;
						}

						if (tn.Output > max)
						{
							max = tn.Output;
							labelVal = tn.LabelVal;
						}
					}
					labels[labelIdx++] = labelVal;

					// skip to the next label
					n += nIdx - 1;
				}
			}
		}
	}
}
