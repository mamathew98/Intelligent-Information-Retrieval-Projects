package com.company;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;
import org.lemurproject.galago.core.parse.Document;
import org.lemurproject.galago.core.parse.stem.Stemmer;
import org.lemurproject.galago.core.retrieval.GroupRetrieval;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.ScoredDocument;
import org.lemurproject.galago.core.retrieval.query.AnnotatedNode;
import org.lemurproject.galago.utility.Parameters;

public class fbData {
    private static final Logger logger = Logger.getLogger("fbData");
    Parameters fbParams;
    List <ScoredDocument> initialResults;
    Set<String> stemmedQueryTerms;
    Retrieval retrieval;
    Set<String> exclusionTerms;
    
    Map <ScoredDocument, Integer> docLength;
    Map<String, Map<ScoredDocument, Integer>> termCounts;
    Map<ScoredDocument, Map<String, Integer>> termCountsReverse;
    
    public fbData(Retrieval r, Set<String> exclusionTerms, List<ScoredDocument> results, Parameters fbParams) throws IOException{
        this.initialResults = results;
        this.fbParams = fbParams;
        this.retrieval = r;
        this.exclusionTerms = exclusionTerms;
        
        docLength = new HashMap <> ();
        termCounts = new HashMap <> ();
        termCountsReverse = new HashMap<>();
        process();
    }

    
    private void process() throws IOException {
        Stemmer stemmer = getStemmer(null, retrieval);

        Map<ScoredDocument, Integer> termCount;
        Document doc;

        Document.DocumentComponents corpusParams = new Document.DocumentComponents(true, false, true);

        String group = fbParams.get("group", (String) null);

        for (ScoredDocument sd : initialResults) {
            if (group != null && retrieval instanceof GroupRetrieval) {
                doc = ((GroupRetrieval) retrieval).getDocument(sd.documentName, corpusParams, group);
            } else {
                doc = retrieval.getDocument(sd.documentName, corpusParams);
            }

            if (doc == null) {
                logger.info("Failed to retrieve document: " + sd.documentName + " -- RM skipping document.");
                continue;
            }

            List<String> docterms = doc.terms;
            docLength.put(sd, doc.terms.size());

            sd.annotation = new AnnotatedNode();
            sd.annotation.extraInfo = "" + docterms.size();

            for (String term : docterms) {
                // perform stopword and query term filtering here 
                String stemmedTerm = (stemmer == null) ? term : stemmer.stem(term);

                if (exclusionTerms.contains(term)) {
                    continue; // on the blacklist
                }
                if (!termCounts.containsKey(term)) {
                    termCounts.put(term, new HashMap<ScoredDocument, Integer>());
                }
                termCount = termCounts.get(term);
                if (termCount.containsKey(sd)) {
                    termCount.put(sd, termCount.get(sd) + 1);
                } else {
                    termCount.put(sd, 1);
                }
            }
        }
        termCountsReverse = convert(termCounts);
    }
    
    
    private Map <ScoredDocument, Map<String, Integer>> convert (Map<String, Map<ScoredDocument, Integer>> in){
        Map <ScoredDocument, Map<String, Integer>> counts = new HashMap <>();
        for (String term : in.keySet()){
            for (ScoredDocument sd : in.get(term).keySet()){
                if (!counts.containsKey(sd)){
                    counts.put(sd, new HashMap <>());
                }
                counts.get(sd).put(term, in.get(term).get(sd));
            }
        }
        return counts;
    }
    
    
    public Map<ScoredDocument, Integer> getDocLength() {
        return docLength;
    }

    public Map<String, Map<ScoredDocument, Integer>> getTermCounts() {
        return termCounts;
    }

    public Map<ScoredDocument, Map<String, Integer>> getTermCountsReverse() {
        return termCountsReverse;
    }


    public List<ScoredDocument> getInitialResults() {
        return initialResults;
    }

    public static Stemmer getStemmer(Parameters p, Retrieval ret) {
        Stemmer stemmer;
        if (ret.getGlobalParameters().isString("rmStemmer")) {
            String rmstemmer = ret.getGlobalParameters().getString("rmStemmer");
            try {
                stemmer = (Stemmer) Class.forName(rmstemmer).getConstructor().newInstance();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } else {
            stemmer = null;
        }
        return stemmer;
    }
}
