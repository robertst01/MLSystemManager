using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Text;
using Newtonsoft.Json;

namespace MLSystemManager.Algorithms
{
	[DataContract]
	public class NeuralNetSave
	{
		[DataMember]
		public double Rate { get; set; }
		[DataMember]
		public double Momentum { get; set; }
		[DataMember]
		public string Activation { get; set; }
		[DataMember]
		public int SnapshotInterval { get; set; }
		[DataMember]
		public int BatchSize { get; set; }
		[DataMember]
		public double ValidationAccuracy { get; set; }
		[DataMember]
		public double Mse { get; set; }
		[DataMember]
		public double InitialMse { get; set; }
		[DataMember]
		public int Epoch { get; set; }
		[DataMember]
		public List<Layer> Layers { get; set; }

		[DataContract]
		public class Node
		{
			[DataMember]
			public int Feature { get; set; }
			[DataMember]
			public bool IsContinuous { get; set; }
			[DataMember]
			public int LabelCol { get; set; }
			[DataMember]
			public double LabelVal { get; set; }
			[DataMember]
			public double[] Weights { get; set; }
		}

		[DataContract]
		public class Layer
		{
			[DataMember]
			public NeuralNet.LayerType Type { get; set; }
			[DataMember]
			public List<Node> Nodes { get; set; }
		}

		private NeuralNetSave()
		{
		}

		public static void Save(string fileName, NeuralNet neuralNet, int epoch, double mse, double initialMse, double accuracy)
		{
			var nns = new NeuralNetSave()
			{
				Rate = neuralNet.Parameters.Rate,
				Momentum = neuralNet.Parameters.Momentum,
				Activation = neuralNet.Parameters.Activation,
				ValidationAccuracy = accuracy,
				SnapshotInterval = neuralNet.Parameters.SnapshotInterval,
				BatchSize = neuralNet.Parameters.BatchSize,
				Mse = mse,
				InitialMse = initialMse,
				Epoch = epoch,
				Layers = new List<Layer>()
			};

			foreach (var layer in neuralNet.Layers)
			{
				var newLayer = new Layer()
				{
					Type = layer.Type,
					Nodes = new List<Node>()
				};

				foreach (var node in layer.Nodes)
				{
					var newNode = new Node()
					{
						Weights = node.Weights
					};

					switch (layer.Type)
					{
						case NeuralNet.LayerType.Input:
							newNode.Feature = ((NeuralNet.InputNode) node).Feature;
							break;

						case NeuralNet.LayerType.Output:
							newNode.IsContinuous = ((NeuralNet.OutputNode) node).IsContinuous;
							newNode.LabelCol = ((NeuralNet.OutputNode) node).LabelCol;
							newNode.LabelVal = ((NeuralNet.OutputNode) node).LabelVal;
							break;
					}

					newLayer.Nodes.Add(newNode);
				}

				nns.Layers.Add(newLayer);
			}

			var json = JsonConvert.SerializeObject(nns, Formatting.Indented);
			using (var w = new StreamWriter(fileName))
			{
				w.Write(json);
			}
		}

		public static void Load(string filePath, NeuralNet neuralNet)
		{
			var r = new StreamReader(filePath);
			var json = r.ReadToEnd();
			var ms = new MemoryStream(Encoding.UTF8.GetBytes(json));
			var ser = new DataContractJsonSerializer(typeof(NeuralNetSave));
			var nns = ser.ReadObject(ms) as NeuralNetSave;
			ms.Close();

			neuralNet.Parameters.Rate = nns.Rate;
			neuralNet.Parameters.Momentum = nns.Momentum;
			neuralNet.Parameters.Activation = nns.Activation;
			neuralNet.Parameters.SnapshotInterval = nns.SnapshotInterval;
			neuralNet.Parameters.BatchSize = nns.BatchSize;
			neuralNet.Parameters.StartEpoch = nns.Epoch;
			neuralNet.Parameters.InitialMse = nns.InitialMse;
			neuralNet.Parameters.StartMse = nns.Mse;

			neuralNet.Layers = new List<NeuralNet.Layer>();
			NeuralNet.Layer prevLayer = null;

			foreach (var layer in nns.Layers)
			{
				var newLayer = new NeuralNet.Layer()
				{
					Type = layer.Type,
					Nodes = new List<NeuralNet.Node>()
				};

				var idx = 0;
				foreach (var node in layer.Nodes)
				{
					NeuralNet.Node newNode = null;
					switch (layer.Type)
					{
						case NeuralNet.LayerType.Input:
							newNode = new NeuralNet.InputNode(idx, node.Feature, null);
							break;

						case NeuralNet.LayerType.Hidden:
							newNode = new NeuralNet.HiddenNode(idx, node.Weights);
							break;

						case NeuralNet.LayerType.Output:
							newNode = new NeuralNet.OutputNode(idx, node.IsContinuous, node.LabelCol, node.LabelVal, node.Weights);
							break;
					}

					idx++;
					newLayer.Nodes.Add(newNode);
				}

				newLayer.Previous = prevLayer;
				neuralNet.Layers.Add(newLayer);

				if (prevLayer != null)
				{
					prevLayer.Next = newLayer;
				}
				prevLayer = newLayer;
			}
		}
	}
}
