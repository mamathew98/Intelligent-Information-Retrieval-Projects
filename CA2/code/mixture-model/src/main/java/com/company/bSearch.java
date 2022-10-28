package com.company;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.RetrievalFactory;
import org.lemurproject.galago.core.retrieval.ScoredDocument;
import org.lemurproject.galago.core.retrieval.prf.ExpansionModel;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;
import static org.lemurproject.galago.core.tools.apps.BatchSearch.logger;
import org.lemurproject.galago.utility.Parameters;

public class bSearch {
    
    public static void main(String[] args) throws Exception{
        // give your index path
        String indexPath = "/home/mamathew/Desktop/UT/Information_Retrieval/HW1/sample";
        // your output
        String outputFileName = "doc_ranking06.txt";
        new bSearch().retrieve(indexPath, outputFileName);
    }

    public void retrieve(String indexPath, String outputFileName) throws Exception {
        int requested = 1000; // number of documents to retrieve
        boolean append = false;
        boolean queryExpansion = true;
        // open index
        Retrieval retrieval = RetrievalFactory.instance(indexPath, Parameters.create());
        
        // load queries
        List <Parameters> queries = new ArrayList <> ();
        // give your query in tsv format
        String inputfilepath = "/home/mamathew/Desktop/topics51-100.tsv";
        BufferedReader reader = new BufferedReader(new FileReader(inputfilepath));
        String line;
        while ((line = reader.readLine()) != null)
        {
            String[] tex = line.split("\t");
            queries.add(Parameters.parseString(String.format("{\"number\":\"%s\", \"text\":\"%s\"}", tex[0], tex[1])));
        }
        reader.close();

//        queries.add(Parameters.parseString(String.format("{\"number\":\"%s\", \"text\":\"%s\"}", "301", "International Organized Crime")));
        
        // open output file
        wResult wResult = new wResult(outputFileName, append);

        // for each query, run it, get the results, print in TREC format
        for (Parameters query : queries) {
            String queryNumber = query.getString("number");
            String queryText = query.getString("text");
            queryText = queryText.toLowerCase(); // option to fold query cases -- note that some parameters may require upper case
            
            logger.info("Processing query #" + queryNumber + ": " + queryText);
            
            query.set("requested", requested);

            Node root = StructuredQuery.parse(queryText);
            Node transformed = retrieval.transformQuery(root, query);
            
            // Query Expansion
            if (queryExpansion){
                // This query expansion technique can be replaced by other approaches.

                // call prf model
//                ExpansionModel qe = new org.lemurproject.galago.core.retrieval.prf.RelevanceModel3(retrieval);

                // call mixture model
                ExpansionModel qe = new fbMixtureModel(retrieval);

                try{
                    query.set("fbOrigWeight", 0.5);
                    query.set("fbTerm", 100.0);
                    Node expandedQuery = qe.expand(root.clone(), query.clone());  
                    transformed = retrieval.transformQuery(expandedQuery, query);
                } catch (Exception ex){
                    ex.printStackTrace();
                }
            }
            
//            System.err.println(transformed.toPrettyString()); // This can be used to print the final query in the Galago language.
            // run query
            List<ScoredDocument> results = retrieval.executeQuery(transformed, query).scoredDocuments;
            
            // print results
            wResult.write(queryNumber, results);
        }
        wResult.close();
    }
}
