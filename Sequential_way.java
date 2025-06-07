import java.io.*;
import java.util.*;

public class Sequential_way {

    static class Movie {
        String title, overview;
        Map<String, Double> tfidf = new HashMap<>();

        Movie(String title, String overview) {
            this.title = title;
            this.overview = overview;
        }
    }

    static List<Movie> readMovies(String filePath) throws IOException {
        List<Movie> movies = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        br.readLine();
        String line;
        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",", 2);
            if (parts.length == 2)
                movies.add(new Movie(parts[0], parts[1]));
        }
        br.close();
        return movies;
    }

    static Map<String, Double> computeIDF(List<Movie> movies) {
        Map<String, Integer> df = new HashMap<>();
        for (Movie m : movies) {
            Set<String> words = new HashSet<>(Arrays.asList(m.overview.split("\\s+")));
            for (String w : words)
                df.put(w, df.getOrDefault(w, 0) + 1);
        }

        int totalDocs = movies.size();
        Map<String, Double> idf = new HashMap<>();
        for (String word : df.keySet())
            idf.put(word, Math.log((double) totalDocs / df.get(word)));

        return idf;
    }

    static Map<String, Double> computeTFIDF(String text, Map<String, Double> idf) {
        Map<String, Integer> tf = new HashMap<>();
        for (String word : text.split("\\s+"))
            tf.put(word, tf.getOrDefault(word, 0) + 1);

        Map<String, Double> tfidf = new HashMap<>();
        for (String word : tf.keySet()) {
            double tfVal = tf.get(word);
            double idfVal = idf.getOrDefault(word, Math.log(1));
            tfidf.put(word, tfVal * idfVal);
        }
        return tfidf;
    }

    static double cosineSimilarity(Map<String, Double> v1, Map<String, Double> v2) {
        Set<String> allWords = new HashSet<>(v1.keySet());
        allWords.addAll(v2.keySet());

        double dot = 0, norm1 = 0, norm2 = 0;
        for (String word : allWords) {
            double a = v1.getOrDefault(word, 0.0);
            double b = v2.getOrDefault(word, 0.0);
            dot += a * b;
            norm1 += a * a;
            norm2 += b * b;
        }

        return (norm1 == 0 || norm2 == 0) ? 0 : dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    static Movie recommend(List<Movie> movies, String input, Map<String, Double> idf) {
        Map<String, Double> inputVec = computeTFIDF(input, idf);
        Movie bestMatch = null;
        double bestScore = -1;

        for (Movie m : movies) {
            double score = cosineSimilarity(inputVec, m.tfidf);
            if (score > bestScore) {
                bestScore = score;
                bestMatch = m;
            }
        }
        return bestMatch;
    }

    public static void main(String[] args) throws IOException {
        List<Movie> movies = readMovies("Movie.csv");
        System.out.println("Loaded " + movies.size() + " movies.");

        Map<String, Double> idf = computeIDF(movies);
        for (Movie m : movies)
            m.tfidf = computeTFIDF(m.overview, idf);

        Scanner sc = new Scanner(System.in);
        System.out.print("Enter keywords: ");
        String input = sc.nextLine();

        long start = System.currentTimeMillis();
        Movie result = recommend(movies, input, idf);
        long end = System.currentTimeMillis();

        System.out.println("\n=== RESULT ===");
        System.out.println("Time: " + (end - start) + " ms");
        System.out.println("Movie: " + result.title);
        System.out.println("Overview: " + result.overview);

        sc.close();
    }
}
