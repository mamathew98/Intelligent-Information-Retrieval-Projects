package com.company;
import javax.annotation.Nonnull;
import org.lemurproject.galago.core.retrieval.prf.WeightedTerm;
import org.lemurproject.galago.utility.lists.Scored;

// This class is copied from RelevanceModel1 in Galago!
public class wordWeight extends WeightedTerm {
    // implementation of weighted term (term, score) pairs
    public String term;

    public wordWeight(String t) {
      this(t, 0.0);
    }

    public wordWeight(String term, double score) {
      super(score);
      this.term = term;
    }

    @Override
    public String getTerm() {
      return term;
    }

    // The secondary sort is to have defined behavior for statistically tied samples.
    @Override
    public int compareTo(@Nonnull WeightedTerm other) {
      wordWeight that = (wordWeight) other;
      int result = -Double.compare(this.score, that.score);
      if (result != 0) {
        return result;
      }
      result = (this.term.compareTo(that.term));
      return result;
    }

    @Override
    public String toString() {
      return "<" + term + "," + score + ">";
    }

    @Override
    public Scored clone(double score) {
      return new wordWeight(this.term, score);
    }
}

