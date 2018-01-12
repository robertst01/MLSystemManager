using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Json;
using System.Text;

namespace MLSystemManager
{
	class ArgParser
	{
	    private Parameters _parameters;

		//You can add more options for specific learning models if you wish
		public ArgParser(string[] argv)
		{
			try
			{
                _parameters = new Parameters();

				for (int i = 0; i < argv.Length; i++)
				{
					if (argv[i] == "-A")
					{
						_parameters.Arff = argv[++i];
					}
					else if (argv[i] == "-B")
					{
						_parameters.Boost = double.Parse(argv[++i]);
					}
					else if (argv[i] == "-AF")
					{
						_parameters.Activation = argv[++i];
						if ((argv[i] == "relu") || (argv[i] == "vrelu"))
						{
							//expecting ReLU leak level or threshold
							_parameters.ActParameter = argv[++i];
						}
					}
					else if (argv[i] == "-C")
					{
						_parameters.CorruptLevel = double.Parse(argv[++i]);
					}
					else if (argv[i] == "-D")
					{
						_parameters.Distance = true;
					}
					else if (argv[i] == "-E")
					{
						_parameters.Evaluation = argv[++i];
						if (argv[i] == "static")
						{
                            //expecting a test set name
						    _parameters.EvalExtra = argv[++i];
						}
						else if (argv[i] == "random")
						{
                            //expecting a double representing the percentage for testing
                            //Note stratification is NOT done
						    _parameters.EvalExtra = argv[++i];
						}
						else if (argv[i] == "cross")
						{
                            //expecting the number of folds
						    _parameters.EvalExtra = argv[++i];
						}
						else if (argv[i] != "training")
						{
							Console.WriteLine("Invalid Evaluation Method: " + argv[i]);
							Environment.Exit(0);
						}
					}
					else if (argv[i] == "-F")
					{
					    _parameters.OutputFileName = argv[++i];
					}
					else if (argv[i] == "-G")
					{
					    _parameters.GridSize = int.Parse(argv[++i]);
					}
					else if (argv[i].StartsWith("-H"))
					{
						string hlNumStr = argv[i];
						int hlNum = 0;
						if ((hlNumStr.Length < 3) || !int.TryParse(hlNumStr.Substring(2, hlNumStr.Length - 2), out hlNum))
						{
							Console.WriteLine("Invalid Hidden Layer: " + hlNumStr);
							Environment.Exit(0);
						}
						if (hlNum > _parameters.Hidden.Count + 1)
						{
							Console.WriteLine("Hidden Layers must be sequential (-H1, -H2, etc.): " + hlNumStr);
							Environment.Exit(0);
						}
						string hlCountStr = argv[++i];
						int hlCount = 0;
						if (!int.TryParse(hlCountStr, out hlCount))
						{
							Console.WriteLine("Invalid Hidden Layer Count: " + hlCountStr);
							Environment.Exit(0);
						}

					    _parameters.Hidden.Add(hlCount);
					}
					else if (argv[i] == "-I")
					{
					    _parameters.Ignore = argv[++i];
					}
					else if (argv[i] == "-IT")
					{
					    _parameters.Iterations = int.Parse(argv[++i]);
					}
					else if (argv[i] == "-K")
					{
					    _parameters.K = int.Parse(argv[++i]);
					}
					else if (argv[i] == "-L")
					{
					    _parameters.Learner = argv[++i];
						if (_parameters.Learner == "clustering")
						{
                            //expecting the type (k, single, complete)
						    _parameters.LearnExtra = argv[++i];
						}
					}
					else if (argv[i] == "-M")
					{
					    _parameters.Momentum = double.Parse(argv[++i]);
					}
					else if (argv[i] == "-N")
					{
					    _parameters.Normalize = true;
					}
					else if (argv[i] == "-NO")
					{
					    _parameters.NormalizeOutputs = true;
					}
					else if (argv[i] == "-O")
					{
					    _parameters.Outputs = int.Parse(argv[++i]);
					}
					else if (argv[i] == "-P")
					{
					    _parameters.Prune = true;
					}
					else if (argv[i] == "-PH")
					{
					    _parameters.Ph = double.Parse(argv[++i]);
					}
					else if (argv[i] == "-PI")
					{
					    _parameters.Pi = double.Parse(argv[++i]);
					}
					else if (argv[i] == "-R")
					{
					    _parameters.Rate = double.Parse(argv[++i]);
					}
					else if (argv[i] == "-S")
					{
					    _parameters.Sample = true;
					}
					else if (argv[i] == "-TA")
					{
					    _parameters.TrainAll = true;
					}
					else if (argv[i] == "-V")
					{
					    _parameters.Verbose = true;
					}
					else if (argv[i] == "-W")
					{
					    _parameters.WeightsFileName = argv[++i];
					}
					else
					{
						Console.WriteLine("Invalid parameter: " + argv[i]);
						Environment.Exit(0);
					}
				}

			}
			catch (Exception e)
			{
				Console.WriteLine(e.Message);
				Console.WriteLine("Usage:");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E [evaluationMethod] {[extraParamters]} [OPTIONS]\n");
				Console.WriteLine("OPTIONS:");
				Console.WriteLine("-V Print the confusion matrix and learner accuracy on individual class values\n");

				Console.WriteLine("Possible evaluation methods are:");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E training");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E static [testARFF_File]");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E random [%_ForTraining]");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]\n");
				Environment.Exit(0);
			}

			if (_parameters.Arff == null || _parameters.Learner == null || _parameters.Evaluation == null)
			{
				Console.WriteLine("Usage:");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E [evaluationMethod] {[extraParamters]} [OPTIONS]\n");
				Console.WriteLine("OPTIONS:");
				Console.WriteLine("-V Print the confusion matrix and learner accuracy on individual class values");
				Console.WriteLine("-N Use normalized data");
				Console.WriteLine();
				Console.WriteLine("Possible evaluation methods are:");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E training");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E static [testARFF_File]");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E random [%_ForTraining]");
				Console.WriteLine("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]\n");
				Environment.Exit(0);
			}

            //Create a stream to serialize the object to.  
		    var ms = new MemoryStream();

		    // Serializer the User object to the stream.  
		    var ser = new DataContractJsonSerializer(typeof(Parameters));
		    ser.WriteObject(ms, _parameters);
		    byte[] json = ms.ToArray();
		    ms.Close();

            var w = new StreamWriter("Parameters.txt");
		    w.Write(Encoding.UTF8.GetString(json, 0, json.Length));
            w.Close();
        }

        //The getter methods
        public string GetARFF() { return _parameters.Arff; }
		public string GetLearner() { return _parameters.Learner; }
		public string GetLearnParameter() { return _parameters.LearnExtra; }
		public string GetEvaluation() { return _parameters.Evaluation; }
		public string GetEvalParameter() { return _parameters.EvalExtra; }
		public double GetRate() { return _parameters.Rate; }
		public double GetMomentum() { return _parameters.Momentum; }
		public double GetPH() { return _parameters.Ph; }
		public double GetPI() { return _parameters.Pi; }
		public int[] GetHidden() { return _parameters.Hidden.ToArray(); }
		public int GetOutputs() { return _parameters.Outputs; }
		public string GetOutputFileName() { return _parameters.OutputFileName; }
		public string GetWeightsFileName() { return _parameters.WeightsFileName; }
		public bool GetVerbose() { return _parameters.Verbose; }
		public bool GetNormalize() { return _parameters.Normalize; }
		public bool GetNormalizeOutputs() { return _parameters.NormalizeOutputs; }
		public bool GetPrune() { return _parameters.Prune; }
		public bool GetDistance() { return _parameters.Distance; }
		public int GetK() { return _parameters.K; }
		public string GetIgnore() { return _parameters.Ignore; }
		public bool GetSample() { return _parameters.Sample; }
		public double GetCorruptLevel() { return _parameters.CorruptLevel; }
		public int GetGridSize() { return _parameters.GridSize; }
		public int GetIterations() { return _parameters.Iterations; }
		public string GetActivation() { return _parameters.Activation; }
		public string GetActParameter() { return _parameters.ActParameter; }
		public bool GetTrainAll() { return _parameters.TrainAll; }
		public double GetBoost() { return _parameters.Boost; }
	}
}
