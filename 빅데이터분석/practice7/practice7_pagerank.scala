import org.apache.spark.graphx.GraphLoader
import java.io._

val graph = GraphLoader.edgeListFile(sc, "practice8_pagerank.txt")

val ranks = graph.pageRank(0.001).vertices
ranks.foreach(println)

val swappedRanks = ranks.map(_.swap)

val sortedRanks = swappedRanks.sortByKey(false)


val pw = new PrintWriter(new File("result.txt"))


for(pr_node <- sortedRanks.collect().take(10)){
    pw.write(pr_node.toString)
    pw.println("\n")
    }



pw.close()
sc.stop()


