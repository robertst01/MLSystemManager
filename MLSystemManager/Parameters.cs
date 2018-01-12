using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLSystemManager
{
    public class Parameters
    {
        public string Arff { get; set; }
        public string Learner { get; set; }
        public string LearnExtra { get; set; }
        public string Evaluation { get; set; }
        public string EvalExtra { get; set; }
        public double Rate { get; set; } = 0.1;
        public double Momentum { get; set; } = 0.9;
        public double Ph { get; set; } = 0.5;
        public double Pi { get; set; } = 0.8;
        public int Outputs { get; set; } = 1;
        public List<int> Hidden { get; set; } = new List<int>();
        public string OutputFileName { get; set; }
        public string WeightsFileName { get; set; }
        public bool Verbose { get; set; } = false;
        public bool Normalize { get; set; } = false;
        public bool NormalizeOutputs { get; set; } = false;
        public bool Prune { get; set; } = false;
        public bool Distance { get; set; } = false;
        public int K { get; set; } = 1;
        public string Ignore { get; set; }
        public bool Sample { get; set; } = false;
        public double CorruptLevel { get; set; } = 0;
        public int GridSize { get; set; } = 1;
        public int Iterations { get; set; } = 10000;
        public string Activation { get; set; } = "sigmoid";
        public string ActParameter { get; set; } = "0,0,1";
        public bool TrainAll { get; set; } = false;
        public double Boost { get; set; } = 1.0;
    }
}
