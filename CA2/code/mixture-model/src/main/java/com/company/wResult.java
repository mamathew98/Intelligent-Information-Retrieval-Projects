package com.company;
import java.io.BufferedOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.util.List;
import org.lemurproject.galago.core.retrieval.ScoredDocument;

public class wResult {
    private PrintStream out;
    
    public wResult (String outputFileName, boolean append) throws UnsupportedEncodingException, FileNotFoundException{
        out = new PrintStream(new BufferedOutputStream(new FileOutputStream(outputFileName, append)), true, "UTF-8");
    }
    
    public wResult (){
        out = System.out;
    }
    
    public void write (String queryNumber, List<ScoredDocument> results, boolean trecFormat) {
        if (!results.isEmpty()) {
            for (ScoredDocument sd : results) {
                if (trecFormat) {
                    out.println(sd.toTRECformat(queryNumber));
                } else {
                    out.println(sd.toString(queryNumber));
                }
            }
        }
    }
    
    public void write (String queryNumber, List<ScoredDocument> results) {
        write(queryNumber, results, true);
    }
    
    public void close (){
        out.close();
    }
}
