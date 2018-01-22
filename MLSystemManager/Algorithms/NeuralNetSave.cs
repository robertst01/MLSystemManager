using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Text;
using System.Threading.Tasks;
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
		public double TrainAccuracy { get; set; }
		[DataMember]
		public double ValidationAccuracy { get; set; }
		[DataMember]
		public double Mse { get; set; }
		[DataMember]
		public int Epoch { get; set; }
		[DataMember]
		public List<Layer> Layers { get; set; }

		[DataContract]
		public class Node
		{
			[DataMember]
			public NeuralNet.LayerType Type { get; set; }
			[DataMember]
			public int Feature { get; set; }
			[DataMember]
			public double MinValue { get; set; }
			[DataMember]
			public double MaxValue { get; set; }
			[DataMember]
			public bool IsContinuous { get; set; }
			[DataMember]
			public int LabelCol { get; set; }
			[DataMember]
			public double LabelVal { get; set; }
			[DataMember]
			public double[] Weights { get; set; }
		}

		public class Layer
		{
			public NeuralNet.LayerType Type { get; set; }
			public List<Node> Nodes { get; set; }
		}

		public NeuralNetSave(NeuralNet neuralNet)
		{
			Rate = neuralNet.Parameters.Rate;
			Momentum = neuralNet.Parameters.Momentum;
			Activation = neuralNet.Parameters.Activation;
			Layers = new List<Layer>();
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
						Type = layer.Type,
						Weights = node.Weights
					};

					switch (layer.Type)
					{
						case NeuralNet.LayerType.Input:
							newNode.Feature = ((NeuralNet.InputNode)node).Feature;
							newNode.MinValue = ((NeuralNet.InputNode)node).MinValue;
							newNode.MaxValue = ((NeuralNet.InputNode)node).MaxValue;
							break;

						case NeuralNet.LayerType.Output:
							newNode.IsContinuous = ((NeuralNet.OutputNode)node).IsContinuous;
							newNode.LabelCol = ((NeuralNet.OutputNode)node).LabelCol;
							newNode.LabelVal = ((NeuralNet.OutputNode)node).LabelVal;
							break;
					}

					newLayer.Nodes.Add(newNode);
				}

				Layers.Add(newLayer);
			}
		}

		public void Save(string fileName)
		{
			var json = JsonConvert.SerializeObject(this, Formatting.Indented);
			using (var w = new StreamWriter(fileName))
			{
				w.Write(json);
			}
		}

		public static NeuralNetSave Load(string filePath)
		{
			var r = new StreamReader(filePath);
			var json = r.ReadToEnd();
			var ms = new MemoryStream(Encoding.UTF8.GetBytes(json));
			var ser = new DataContractJsonSerializer(typeof(NeuralNetSave));
			var neuralNetSave = ser.ReadObject(ms) as NeuralNetSave;
			ms.Close();

			return neuralNetSave;
		}
	}
}
