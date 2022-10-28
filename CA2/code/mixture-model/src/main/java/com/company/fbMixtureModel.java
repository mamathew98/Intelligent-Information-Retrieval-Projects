
package com.company;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;

import javax.annotation.Nonnull;

import org.lemurproject.galago.core.index.stats.FieldStatistics;
import org.lemurproject.galago.core.index.stats.NodeStatistics;
import org.lemurproject.galago.core.parse.stem.Stemmer;
import org.lemurproject.galago.core.retrieval.Results;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.ScoredDocument;
import org.lemurproject.galago.core.retrieval.prf.ExpansionModel;
import org.lemurproject.galago.core.retrieval.prf.WeightedTerm;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;
import org.lemurproject.galago.core.util.WordLists;
import org.lemurproject.galago.utility.Parameters;
import org.lemurproject.galago.utility.lists.Scored;

public class fbMixtureModel implements ExpansionModel {
    protected Retrieval retrieval;
    int defaultFbDocs, defaultFbTerms;
    double defaultFbOrigWeight;
    Set<String> exclusionTerms;
    Stemmer stemmer;

    public fbMixtureModel(Retrieval r) throws IOException {
        retrieval = r;
        defaultFbDocs = (int) Math.round(r.getGlobalParameters().get("fbDocs", 15.0));
        defaultFbTerms = (int) Math.round(r.getGlobalParameters().get("fbTerm", 500.0));
        defaultFbOrigWeight = r.getGlobalParameters().get("fbOrigWeight", 0.2);
        exclusionTerms = WordLists.getWordList(r.getGlobalParameters().get("rmstopwords", "rmstop"));
        Parameters gblParms = r.getGlobalParameters();
        this.stemmer = fbData.getStemmer(gblParms, retrieval);
    }

    public List<ScoredDocument> collectInitialResults(Node transformed, Parameters fbParams) throws Exception {
        Results results = retrieval.executeQuery(transformed, fbParams);
        List<ScoredDocument> res = results.scoredDocuments;
        if (res.isEmpty())
            throw new Exception("No feedback documents found!");
        return res;
    }

    public Node generateExpansionQuery(List<WeightedTerm> weightedTerms, int fbTerms) throws IOException, Exception {
        Node expNode = new Node("combine");
//        System.err.println("Feedback Terms:");
        for (int i = 0; i < Math.min(weightedTerms.size(), fbTerms); i++) {
            Node expChild = new Node("text", weightedTerms.get(i).getTerm());
            expNode.addChild(expChild);
            expNode.getNodeParameters().set("" + i, weightedTerms.get(i).getWeight());
        }
        return expNode;
    }

    public int getFbDocCount(Node root, Parameters queryParameters) throws Exception {
        int fbDocs = (int) Math.round(root.getNodeParameters().get("fbDocs", queryParameters.get("fbDocs", (double) defaultFbDocs)));
        if (fbDocs <= 0)
            throw new Exception("Invalid number of feedback documents!");
        return fbDocs;
    }

    public int getFbTermCount(Node root, Parameters queryParameters) throws Exception {
        int fbTerms = (int) Math.round(root.getNodeParameters().get("fbTerm", queryParameters.get("fbTerm", (double) defaultFbTerms)));
        if (fbTerms <= 0)
            throw new Exception("Invalid number of feedback terms!");
        return fbTerms;
    }

    public Node interpolate(Node root, Node expandedQuery, Parameters queryParameters) throws Exception {
        queryParameters.set("defaultFbOrigWeight", defaultFbOrigWeight);
        queryParameters.set("fbOrigWeight", queryParameters.get("fbOrigWeight", defaultFbOrigWeight));
        return linearInterpolation(root, expandedQuery, queryParameters);
    }

    public Node linearInterpolation(Node root, Node expNode, Parameters parameters) throws Exception {
        double defaultFbOrigWeight = parameters.get("defaultFbOrigWeight", -1.0);
        if (defaultFbOrigWeight < 0)
            throw new Exception("There is not defaultFbOrigWeight parameter value");
        double fbOrigWeight = parameters.get("fbOrigWeight", defaultFbOrigWeight);
        if (fbOrigWeight == 1.0) {
            return root;
        }
        Node result = new Node("combine");
        result.addChild(root);
        result.addChild(expNode);
        result.getNodeParameters().set("0", fbOrigWeight);
        result.getNodeParameters().set("1", 1.0 - fbOrigWeight);
        return result;
    }

    public Parameters getFbParameters(Node root, Parameters queryParameters) throws Exception {
        Parameters fbParams = Parameters.create();
        fbParams.set("requested", getFbDocCount(root, queryParameters));
        fbParams.set("passageQuery", false);
        fbParams.set("extentQuery", false);
        fbParams.setBackoff(queryParameters);
        return fbParams;
    }

    @Override
    public Node expand(Node root, Parameters queryParameters) throws Exception {
        int fbTerms = getFbTermCount(root, queryParameters);
        // transform query to ensure it will run
        Parameters fbParams = getFbParameters(root, queryParameters);
        Node transformed = retrieval.transformQuery(root.clone(), fbParams);

        // get some initial results
        List<ScoredDocument> initialResults = collectInitialResults(transformed, fbParams);


        // extract grams from results
        Set<String> queryTerms = getTerms(stemmer, StructuredQuery.findQueryTerms(transformed));
        fbData fbData = new fbData(retrieval, exclusionTerms, initialResults, fbParams);
        List<WeightedTerm> weightedTerms = computeWeights(fbData, fbParams, queryParameters);
        Collections.sort(weightedTerms);
        Node expNode = generateExpansionQuery(weightedTerms, fbTerms);

        return interpolate(root, expNode, queryParameters);
    }

    public static Set<String> getTerms(Stemmer stemmer, Set<String> terms) {
        if (stemmer == null)
            return terms;

        Set<String> stems = new HashSet<String>(terms.size());
        for (String t : terms) {
            String s = stemmer.stem(t);
            stems.add(s);
        }
        return stems;
    }

    //computeWeights function returns a list of terms with their weights extracted from the feedback docs
	// This part does the EM step of the mixture model
    public List<WeightedTerm> computeWeights(fbData fbData, Parameters fbParam, Parameters queryParameters) throws Exception {
        try {
        	// p(w | thetaF)
            HashMap<String, Double> p_ThetaF = new HashMap<>();
            
            // p(w | C)
            HashMap<String, Double> p_ThetaC = new HashMap<>();
            
            // p (z==1 | w)
            HashMap<String, Double> p_Zt = new HashMap<>();
            
            // feedback terms data
            Map<String, Map<ScoredDocument, Integer>> termCounts = fbData.termCounts;
            
            // feedback terms
            HashMap<String,Integer> feedback_term_count = new HashMap<>();

//            Set<String> queryTerms = fbData.stemmedQueryTerms;
            Set<String> queryTerms = new HashSet<>();
            
            // get query terms and feedback #term info
            for(Map.Entry<String, Map<ScoredDocument, Integer>> t: termCounts.entrySet()){
            	queryTerms.add(t.getKey());
            	
            	
            	// old way of getting feedback term count that I think is somehow wrong
            	// ======================================================================
//            	if (feedback_term_count.get(t.getKey())!=null){
//            		feedback_term_count.put(t.getKey(),feedback_term_count.get(t.getKey())+1);
//            	}else {
//            		feedback_term_count.put(t.getKey(),1);
//            	}
            	//=======================================================================
            	
            	/////////////////////////////////////////////////////////////////////////////////
            	
            	// new way of getting feedback term counts that I came up with
            	//=======================================================================
            	feedback_term_count.put(t.getKey(), 0);
            	for (Map.Entry<ScoredDocument, Integer> t2: t.getValue().entrySet()) {
            		feedback_term_count.put(t.getKey(),feedback_term_count.get(t.getKey())+t2.getValue());
            	}	
            	//=======================================================================
            }
            
            Set<String> excTerms = fbData.exclusionTerms;
            double lambda = 1.0 - fbParam.getDouble("fbOrigWeight");
            
            //get corpus length
            Retrieval r = fbData.retrieval;
            Node n = new Node();
            n.setOperator("lengths");
            n.getNodeParameters().set("part", "lengths");
            FieldStatistics stat = retrieval.getCollectionStatistics(n);
            double corpusLen = stat.documentCount;
            
            //removing exclusions
            if(excTerms!=null) queryTerms.removeAll(excTerms);
            
            //initialize p_ThetaF (all terms equal)
            for(String s : queryTerms){
                p_ThetaF.put(s, 1.0/queryTerms.size());
            }
            
            //calculate p_ThetaC
            for(String s : queryTerms){
                double cT_F = termFreqInCorpus(s, corpusLen, retrieval);
                p_ThetaC.put(s, cT_F);
            }
            List<WeightedTerm> wt = new ArrayList<>();
            
       		DecimalFormat df = new DecimalFormat("#");
            df.setMaximumFractionDigits(8);

            // put  EM code here
            //==========================
            boolean converged = false;  
            int loop = 1;
//            while (loop <= 2) { // for test purpose
            while (!converged) {
            	for(String s : queryTerms) {
            		
             	   //            [           lambda * p(w|C)            ]
             	   // p(z=1|w) = [ ------------------------------------ ]
             	   //            [ lambda * p(w|C) = (1-lambda) * p(w|F)]
            		
               		double numerator = lambda * p_ThetaC.get(s);
//               		System.out.println("ZNumerator: " + numerator);
               		double denominator = (lambda * p_ThetaC.get(s)) + ((1 - lambda) * p_ThetaF.get(s));
//               		System.out.println("ZDenominator: " + denominator);
               		double newZ = (numerator / denominator);
               		p_Zt.put(s, newZ);
               }
               
            	// calculate sigma for normalization
               double sigma = 0;
               for (String s : queryTerms) {
               		sigma += feedback_term_count.get(s) * (1 - p_Zt.get(s));
               }
               
               // define convergence thershold
               double thersshold = 0.00001;
               
               converged = true;
               for (String s : queryTerms) {
            	   
            	   //             [     c(w|F) * (1 - p(z=1|w))    ]
            	   // Pn+1(w|F) = [ -------------------------------]
            	   //             [ Sigma{c(w|F) * (1 - p(z=1|w))} ]
            	   
               		double numerator2 = feedback_term_count.get(s) * (1 - p_Zt.get(s));
//               		System.out.println("FNumerator: " + numerator);
               		double newThetaF = (numerator2 / sigma); 
//               		System.out.println("FNew: " + newThetaF);
               		if(p_ThetaF.get(s) != null && (Math.abs(p_ThetaF.get(s) - newThetaF)) > thersshold) {
               			converged = false;
               		}
               		p_ThetaF.put(s, newThetaF);
               } 
//               loop++; // for test purpose
           }
            
            for(String s : queryTerms) {
//            	wordWeight term = new wordWeight(s, Math.log(p_ThetaF.get(s))); // this is log-weight
            	wordWeight term = new wordWeight(s,p_ThetaF.get(s)); // this is normal weight
            	System.out.println(term);
            	wt.add(term);
            }
            return wt;
        }
        catch (Exception e) {
//        	System.out.println(e.toString());
            e.printStackTrace();
            throw new Exception("This should be implemented! This method outputs a list of terms with weights.");
        }
    }

    private double termFreqInCorpus(String s, double corpusLen, Retrieval retrieval){
        try {
            String que = s;
            Node node = StructuredQuery.parse(que);
            node.getNodeParameters().set("queryType", "count");
            node = retrieval.transformQuery(node, Parameters.create());
            NodeStatistics stat2 = retrieval.getNodeStatistics(node);
            return stat2.nodeDocumentCount/corpusLen;
        }catch (Exception e){
            e.printStackTrace();
            throw new NoSuchElementException();
        }
    }

    private double termFreqInRel(String s, fbData fbData){
        Map<String, Map<ScoredDocument, Integer>> termCounts = fbData.termCounts;
        Map<ScoredDocument, Integer> map = termCounts.get(s);
        double ans = 0.0;
        for(Map.Entry<ScoredDocument, Integer> m : map.entrySet()){
            ans += m.getValue();
        }
        return ans;
    }
    
}
