// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

/**
 * For nominal labels, this model simply returns the majority class. For
 * continuous labels, it returns the mean value.
 * If the learning model you're using doesn't do as well as this one,
 * it's time to find a new learning model.
 */
namespace MLSystemManager
{
	public class BaselineLearner : SupervisedLearner
	{
		private double[] m_labels;

		public override void Train(Matrix features, Matrix labels, double[] colMin, double[] colMax)
		{
			m_labels = new double[labels.Cols()];
			for (int i = 0; i < labels.Cols(); i++)
			{
				if (labels.ValueCount(i) == 0)
					m_labels[i] = labels.ColumnMean(i); // continuous
				else
					m_labels[i] = labels.MostCommonValue(i); // nominal
			}
		}

		public override void VTrain(VMatrix features, VMatrix labels, double[] colMin, double[] colMax)
		{
		}

		public override void Predict(double[] features, double[] labels)
		{
			for (int i = 0; i < m_labels.Length; i++)
				labels[i] = m_labels[i];
		}
	}
}